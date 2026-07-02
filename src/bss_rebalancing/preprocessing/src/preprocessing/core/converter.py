"""
Generalized trip-data converter.

Takes whatever raw CSVs a source's ZIP(s) extracted into a trips directory,
concatenates them, renames/prunes columns per the source's column_map,
encodes station ids (numeric or alphanumeric scheme — declared per source,
sanity-checked at runtime because data providers lie), and writes ONE clean
CSV. Deletes the loose raw CSVs afterward. ZIPs and the final CSV survive;
nothing else does.
"""

import re
from pathlib import Path
from typing import List

import polars as pl
from tqdm import tqdm

from preprocessing.core.sources import Source

TIMESTAMP_FORMATS = [
    "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %I:%M %p",
    "%m/%d/%y %I:%M:%S %p", "%m/%d/%y %I:%M %p",
    "%m/%d/%Y %H:%M:%S%.f", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y",
    "%m-%d-%Y %H:%M:%S", "%m-%d-%Y",
    "%d.%m.%Y %H:%M:%S%.f", "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%d.%m.%Y",
    "%d.%m.%y %H:%M:%S%.f", "%d.%m.%y %H:%M:%S", "%d.%m.%y",
    "%d/%m/%Y %H:%M:%S%.f", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y",
    "%d/%m/%y %H:%M:%S%.f", "%d/%m/%y %H:%M:%S", "%d/%m/%y",
    "%d-%m-%Y %H:%M:%S%.f", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%d-%m-%Y",
    "%d-%m-%y %H:%M:%S%.f", "%d-%m-%y %H:%M:%S", "%d-%m-%y",
    "%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S%.f", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d",
    "%Y.%m.%d %H:%M:%S%.f", "%Y.%m.%d %H:%M:%S", "%Y.%m.%d",
    "%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%.fZ",
    "%d %b %Y %H:%M:%S", "%d %b %Y", "%d %B %Y %H:%M:%S", "%d %B %Y",
    "%b %d, %Y %H:%M:%S", "%b %d, %Y", "%B %d, %Y",
]


def parse_datetime_multi(col: pl.Expr) -> pl.Expr:
    normalized = col.str.strip_chars().str.replace_all(r"\s+", " ")
    attempts = [normalized.str.to_datetime(fmt, strict=False) for fmt in TIMESTAMP_FORMATS]
    return pl.coalesce(attempts)


def _looks_numeric(sid: str) -> bool:
    return bool(re.match(r'^\d+(\.\d+)?$', sid.strip()))


def encode_station_id_alnum(sid: str) -> int:
    sid = re.sub(r'[^A-Za-z0-9.]', '', sid.strip().upper())
    m = re.match(r'^([A-Z]*)(\d+)(?:\.(\d+))?$', sid)
    if not m:
        raise ValueError(f"Unrecognized station id format: {sid!r}")
    letters, intpart, decpart = m.groups()
    decpart = (decpart or "0").ljust(2, "0")[:2]
    if letters:
        letter_code = "".join(f"{ord(c) - 64:02d}" for c in letters)
        return int("2" + letter_code + intpart + decpart)
    return int("1" + intpart + decpart)


def encode_station_id_numeric(sid: str) -> int:
    # BlueBikes ids are already plain numbers (e.g. '74.0'). Scale by 100
    # so the encoding lands in the same integer space as the alnum scheme —
    # downstream code doesn't need to know or care which source it came from.
    return int(round(float(sid.strip()) * 100))


class TripDataConverter:
    """Concatenates + cleans raw trip CSVs for one Source into one CSV."""

    def __init__(self, source: Source):
        self.source = source

    def _resolve_column_map(self) -> dict:
        return {k: v for k, v in self.source.column_map.items() if v and str(v).strip()}

    def _encode_station_id(self, sid: str) -> int:
        if self.source.station_id_type == "numeric":
            return encode_station_id_numeric(sid)
        return encode_station_id_alnum(sid)

    def _sanity_check_id_type(self, unique_ids: List[str]) -> None:
        """Declared type vs. actual data can drift — providers change
        formats without telling anyone. Warn, don't crash."""
        sample = unique_ids[:50]
        frac_numeric = sum(_looks_numeric(s) for s in sample) / max(len(sample), 1)
        declared_numeric = self.source.station_id_type == "numeric"
        if declared_numeric and frac_numeric < 0.9:
            print(f"WARNING [{self.source.id}]: declared 'numeric' but only "
                  f"{frac_numeric:.0%} of sampled ids look numeric. Check sources.json.")
        if not declared_numeric and frac_numeric > 0.95:
            print(f"NOTE [{self.source.id}]: declared 'alphanumeric' but "
                  f"{frac_numeric:.0%} of sampled ids are plain numbers. Might be fine.")

    def convert(self, trips_dir: str, output_file: str, cleanup: bool = True) -> str:
        trips_dir = Path(trips_dir)
        raw_files = sorted(
            str(p) for p in trips_dir.glob("*.csv")
            if p.resolve() != Path(output_file).resolve()
        )
        if not raw_files:
            raise FileNotFoundError(f"No raw CSVs found in {trips_dir}.")

        print(f"[{self.source.id}] Found {len(raw_files)} raw CSV(s). Concatenating...")
        forced_string_cols = list(self.source.column_map.keys())
        dfs, failed = [], []
        for f in tqdm(raw_files):
            try:
                dfs.append(pl.read_csv(
                    f, infer_schema_length=10000, try_parse_dates=False,
                    schema_overrides={c: pl.Utf8 for c in forced_string_cols},
                ))
            except Exception as e:
                print(f"  ! Skipping {f}: {e}")
                failed.append(f)
        if failed:
            print(f"HEADS UP: {len(failed)}/{len(raw_files)} files failed and were excluded.")
        if not dfs:
            raise RuntimeError(f"[{self.source.id}] Every raw file failed to read.")

        combined = pl.concat(dfs, how="diagonal_relaxed")

        rename_map = self._resolve_column_map()
        rename_map = {k: v for k, v in rename_map.items() if k in combined.columns}
        if rename_map:
            combined = combined.rename(rename_map)

        # Only keep what's mapped/declared vital. Everything else is dead
        # weight — you said it yourself, MB wasted for nothing.
        keep_cols = list(dict.fromkeys(list(rename_map.values()) + self.source.vital_columns))
        keep_cols = [c for c in keep_cols if c in combined.columns]
        combined = combined.select(keep_cols)

        time_col, end_time_col = "starttime", "stoptime"
        for col in (time_col, end_time_col):
            if col not in combined.columns:
                raise KeyError(f"'{col}' missing after rename/select. Have: {combined.columns}")

        combined = combined.with_columns([
            parse_datetime_multi(pl.col(time_col)).alias(time_col),
            parse_datetime_multi(pl.col(end_time_col)).alias(end_time_col),
        ]).sort(time_col)

        for col in (time_col, end_time_col):
            nulls = combined[col].null_count()
            if nulls:
                print(f"WARNING: {nulls} rows failed to parse '{col}'.")

        combined = combined.with_columns([
            pl.col(time_col).dt.strftime("%Y-%m-%d %H:%M:%S%.3f"),
            pl.col(end_time_col).dt.strftime("%Y-%m-%d %H:%M:%S%.3f"),
        ])

        before = combined.height
        vital_present = [c for c in self.source.vital_columns if c in combined.columns]
        combined = combined.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in vital_present]))
        if before - combined.height:
            print(f"Dropped {before - combined.height}/{before} rows missing vital fields.")

        id_cols = [c for c in ("start station id", "end station id") if c in combined.columns]
        if id_cols:
            unique_ids = pl.concat([combined[c] for c in id_cols]).drop_nulls().cast(pl.Utf8).unique().to_list()
            self._sanity_check_id_type(unique_ids)

            id_map, failed_ids = {}, []
            for sid in unique_ids:
                try:
                    id_map[sid] = self._encode_station_id(sid)
                except ValueError as e:
                    failed_ids.append(sid)
                    print(f"  ! Could not encode station id: {e}")

            if failed_ids:
                before_drop = combined.height
                combined = combined.filter(
                    pl.all_horizontal([~pl.col(c).cast(pl.Utf8).is_in(failed_ids) for c in id_cols])
                )
                print(f"Dropped {before_drop - combined.height} rows with unencodable id(s): {failed_ids}")

            combined = combined.with_columns([
                pl.col(c).cast(pl.Utf8).replace_strict(id_map, default=None, return_dtype=pl.Int64)
                for c in id_cols
            ])

        print(f"[{self.source.id}] Final shape: {combined.shape}")
        combined.write_csv(output_file)
        print(f"Written to {output_file}")

        if cleanup:
            for f in raw_files:
                Path(f).unlink(missing_ok=True)
            print(f"Cleaned up {len(raw_files)} raw CSV(s). Kept ZIPs + {output_file}.")

        return output_file