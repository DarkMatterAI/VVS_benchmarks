"""
reaction_assembly.py
────────────────────────────────────────────────────────────────
Efficient parallel reaction enumeration for BB-pairs.

* React every *unique* (bb1, bb2) only once (order-independent).
* LRU cache for pair → product_list in the main process.
* Persistent Pool (optionally) that is recycled every *refresh_every* calls.
* Configurable: list of SMARTS templates + “add explicit Hs” flag.
"""

from __future__      import annotations
from collections     import OrderedDict
from multiprocessing import Pool
from typing          import Dict, List, Tuple, Sequence, Optional
from rdkit           import Chem
from rdkit.Chem      import AllChem
from tqdm            import tqdm

# ────────────────────────────────────────────────────────────────────
# Pool-worker globals + initialiser
# ────────────────────────────────────────────────────────────────────
_WORKER_RXNS: Optional[List[Tuple[AllChem.ChemicalReaction,
                                  Tuple[Chem.Mol, Chem.Mol]]]] = None
_WORKER_ADD_HS: bool = True


def _pool_init(reactions, add_hs: bool):
    """Executed *once* in every child process."""
    global _WORKER_RXNS, _WORKER_ADD_HS
    _WORKER_RXNS  = reactions
    _WORKER_ADD_HS = add_hs


def _react_pair_worker(pair: Tuple[str, str]) -> List[str]:
    """Worker-side reaction enumeration using globals set by `_pool_init`."""
    bb1, bb2 = pair
    m1 = Chem.MolFromSmiles(bb1)
    m2 = Chem.MolFromSmiles(bb2)
    if _WORKER_ADD_HS:
        m1, m2 = Chem.AddHs(m1), Chem.AddHs(m2)

    prods: set[str] = set()
    for rxn, (r1, r2) in _WORKER_RXNS:                 # A+B & B+A
        for a, b in ((m1, m2), (m2, m1)):
            if a.HasSubstructMatch(r1) and b.HasSubstructMatch(r2):
                for plist in rxn.RunReactants((a, b)):
                    for p in plist:
                        prods.add(Chem.MolToSmiles(Chem.RemoveHs(p)))
    return pair, list(prods)


# ────────────────────────────────────────────────────────────────────
class ReactionAssembly:
    """
    Parallel reaction enumerator with LRU-cache.

    Parameters
    ----------
    rxn_smarts   : Sequence[str]
        SMARTS strings of *two-component* reaction templates.

    num_proc     : int, default 32
        Worker processes; when `≤1` **no** pool is created (serial mode).

    add_hs       : bool, default True
        Whether to call ``Chem.AddHs`` on reactants before matching;
        products are always returned *without* explicit Hs.

    cache_size   : int, default 50_000
        Maximum number of cached (bb1, bb2) → product-list entries.

    refresh_every: int, default 200
        Recycle the multiprocessing pool after this many `.react(...)`
        calls to mitigate memory/fd growth.
    """

    # ────────────────────────────────────────────────────────────────
    def __init__(
        self,
        rxn_smarts: Sequence[str],
        *,
        num_proc: int = 32,
        add_hs: bool = True,
        cache_size: int = 50_000,
        refresh_every: int = 200,
    ):
        # Prepare reaction objects once in the main process -------------
        self.reactions: List[Tuple[AllChem.ChemicalReaction,
                                   Tuple[Chem.Mol, Chem.Mol]]] = []
        for sm in rxn_smarts:
            rxn = AllChem.ReactionFromSmarts(sm)
            rxn.Initialize()
            self.reactions.append((rxn, tuple(rxn.GetReactants())))

        # Config
        self.num_proc      = max(1, num_proc)
        self.add_hs        = add_hs
        self.cache_size    = cache_size
        self.refresh_every = max(1, refresh_every)

        # Internal state
        self._cache: OrderedDict[Tuple[str, str], List[str]] = OrderedDict()
        self._call_count = 0
        self._pool: Pool | None = None

    # ────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────
    def react(
        self,
        idxs:  List[int],
        pairs: List[Tuple[str, str]],
        *,
        show_pbar: bool = False,
    ) -> List[Dict[str, str]]:
        """
        Enumerate reactions for each *bb-pair* in **pairs**.

        Parameters
        ----------
        idxs   : list[int]
            Arbitrary identifiers - forwarded unchanged to the output rows.
        pairs  : list[(bb1, bb2)]
            Reactant SMILES tuples (order ignored, deduplicated internally).

        Returns
        -------
        list[{"idx": int, "result": str}]
            One dict per **product** molecule.
        """
        if len(idxs) != len(pairs):
            raise ValueError("`idxs` and `pairs` must be the same length")

        # 1. canonicalise and collect uniques ---------------------------
        keys = []
        uncached = []
        for p in pairs:
            key = tuple(sorted(p))
            keys.append(key)
            if key not in self._cache:
                uncached.append(key)

        # 2. compute missing reactions ---------------------------------
        if uncached:
            new_map = self._compute_pairs(uncached, show_pbar=show_pbar)
            # Fill / refresh cache (LRU)
            for k, prods in new_map.items():
                if k in self._cache:
                    self._cache.move_to_end(k)
                self._cache[k] = prods
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)

        # 3. scatter to output rows ------------------------------------
        out: List[Dict[str, str]] = []
        for idx, key in zip(idxs, keys):
            for prod in self._cache.get(key, ()):
                out.append({"idx": idx, "result": prod})
        return out

    # ────────────────────────────────────────────────────────────────
    # Internals
    # ────────────────────────────────────────────────────────────────
    def _compute_pairs(
        self,
        uniq_pairs: List[Tuple[str, str]],
        *,
        show_pbar: bool = False,
    ) -> Dict[Tuple[str, str], List[str]]:
        """Compute products *only* for the unique, uncached pairs."""
        if self.num_proc <= 1:           # ── serial fallback
            iterator = map(self._react_pair_local, uniq_pairs)
            if show_pbar:
                iterator = tqdm(iterator, total=len(uniq_pairs),
                                desc="Reacting (serial)")
            results = {}
            for key, products in iterator:
                results[key] = products 
            return results 

        # multiprocessing branch ---------------------------------------
        if self._pool is None:
            self._pool = Pool(
                processes=self.num_proc,
                initializer=_pool_init,
                initargs=(self.reactions, self.add_hs),
            )

        self._call_count += 1
        if self._call_count % self.refresh_every == 0:
            self._recycle_pool()
            

        iterator = self._pool.imap_unordered(_react_pair_worker, uniq_pairs, chunksize=8)
        if show_pbar:
            iterator = tqdm(iterator, total=len(uniq_pairs),
                            desc="Reacting (pool)")
            
        results = {}
        for key, products in iterator:
            results[key] = products
        return results 
    
    # ------------------------------------------------------------------
    def _react_pair_local(self, pair: Tuple[str, str]) -> List[str]:
        """Serial-mode implementation (shares code with worker version)."""
        bb1, bb2 = pair
        m1 = Chem.MolFromSmiles(bb1)
        m2 = Chem.MolFromSmiles(bb2)
        if self.add_hs:
            m1, m2 = Chem.AddHs(m1), Chem.AddHs(m2)

        prods: set[str] = set()
        for rxn, (r1, r2) in self.reactions:
            for a, b in ((m1, m2), (m2, m1)):
                if a.HasSubstructMatch(r1) and b.HasSubstructMatch(r2):
                    for plist in rxn.RunReactants((a, b)):
                        for p in plist:
                            prods.add(Chem.MolToSmiles(Chem.RemoveHs(p)))
        return pair, list(prods)

    # ------------------------------------------------------------------
    def _recycle_pool(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = Pool(
                processes=self.num_proc,
                initializer=_pool_init,
                initargs=(self.reactions, self.add_hs),
            )

    # ─────────────────────── context-manager helpers ─────────────────
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
