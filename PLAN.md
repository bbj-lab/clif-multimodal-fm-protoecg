# Extraction Performance Audit

## Issues Found (ordered by impact)

### 1. CRITICAL: `load_for_hospitalizations` reconstitutes partitions via concat (loader.py)
When `profile_codes`, `compute_bin_edges`, or `build_all_base_events` call
`loader.load_for_hospitalizations("vitals", all_hosp_ids)` with 546K IDs, the code:
```python
parts = [partition[hid] for hid in hospitalization_ids if hid in partition]
return pl.concat(parts)
```
This concatenates **546K tiny DataFrames** — catastrophically slow.
The cached full DataFrame already exists in `self._cache["vitals"]`.

**Fix**: When requested IDs are a large fraction of partitioned data, filter
the cached DataFrame instead. When IDs cover ALL data, return the cache directly.

### 2. HIGH: Post-processing loop loads code_status wrong (cli.py:210-214)
```python
cs = loader.load("code_status")  # full table, every iteration
tables["clif_code_status"] = cs.filter(pl.col("patient_id") == pid)
```
546K iterations × full-table filter = disaster.
code_status is already preloaded and partitioned by hospitalization_id.

**Fix**: Use `loader.load_for_hospitalizations("code_status", [hid])`.

### 3. HIGH: Post-processing loop filters hosp_df per iteration (cli.py:215)
```python
tables["clif_hospitalization"] = hosp_df.filter(hosp_df["hospitalization_id"] == hid)
```
546K iterations × scan through 546K-row DataFrame.

**Fix**: Build a dict of single-row DataFrames from `hosp_lookup` once, look up per hid.

### 4. MEDIUM: Redundant `get_hospitalization_ids` when n_patients=None (cli.py:90-91)
When processing all patients (no `-n` flag), `get_hospitalization_ids(all_patients)`
filters 546K rows with `is_in` on 223K patient IDs — effectively a no-op.

**Fix**: Skip when n_patients is None; just get all hosp IDs directly.

### 5. MEDIUM: `IDMapper.from_clif_tables` uses Python iter_rows (id_mapper.py:71-85)
Iterates 546K rows converting to Python dicts to extract IDs.

**Fix**: Use Polars native operations.

### 6. LOW: `sort_priority()` does linear scan (constants.py:83-88)
Called on every event during sorts. Linear scan through ~25 prefixes per call.

**Fix**: Build a compiled lookup with longest-prefix-match caching.
