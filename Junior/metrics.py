"""Metrics."""

from typing import Any, Dict, Union, List
from dataclasses import dataclass
import datetime

import pandas as pd
import pyspark.sql as ps
from pyspark.sql.functions import col, when, isnan, to_date


@dataclass
class Metric:
    """Base class for Metric"""

    def __call__(self, df: Union[pd.DataFrame, ps.DataFrame]) -> Dict[str, Any]:
        if isinstance(df, pd.DataFrame):
            return self._call_pandas(df)

        if isinstance(df, ps.DataFrame):
            return self._call_pyspark(df)

        msg = (
            f"Not supported type of arg 'df': {type(df)}. "
            "Supported types: pandas.DataFrame, "
            "pyspark.sql.dataframe.DataFrame"
        )
        raise NotImplementedError(msg)

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        return {}


@dataclass
class CountTotal(Metric):
    """Total number of rows in DataFrame"""

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {"total": len(df)}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        return {"total": df.count()}


@dataclass
class CountZeros(Metric):
    """Number of zeros in choosen column"""

    column: str

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = sum(df[self.column] == 0)
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import col, count

        n = df.count()
        k = df.filter(col(self.column) == 0).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountNull(Metric):
    """Number of empty values in choosen columns"""

    columns: List[str]
    aggregation: str = "any"  # either "all", or "any"

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        if self.aggregation == "any":
            k = df[self.columns].isnull().any(axis=1).sum()
        elif self.aggregation == "all":
            k = df[self.columns].isnull().all(axis=1).sum()

        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()
        if self.aggregation == "any":
            # Create a condition that checks if any of the specified columns is null
            condition = when(
                col(self.columns[0]).isNull() | isnan(self.columns[0]),
                True
            )
            for column in self.columns[1:]:
                condition = condition | when(col(column).isNull() | isnan(column), True)

            # Count the rows where the condition is true
            k = df.withColumn("null_condition", condition).filter(col("null_condition")).count()
        elif self.aggregation == "all":
            # Create a condition that checks if all of the specified columns are null
            condition = when(
                col(self.columns[0]).isNull() | isnan(self.columns[0]),
                True
            )
            for column in self.columns[1:]:
                condition = condition & when(col(column).isNull() | isnan(column), True)

            # Count the rows where the condition is true
            k = df.withColumn("null_condition", condition).filter(col("null_condition")).count()

        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountDuplicates(Metric):
    """Number of duplicates in choosen columns"""

    columns: List[str]

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        duplicates = df[df.duplicated(subset=self.columns)]
        n = len(df)
        k = len(duplicates)

        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()
        # Calculate the count of duplicates for PySpark DataFrame
        duplicate_df = df.groupBy(self.columns).count()
        k = duplicate_df.filter(duplicate_df['count'] > 1).count()

        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountValue(Metric):
    """Number of values in choosen column"""

    column: str
    value: Union[str, int, float]

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = (df[self.column] == self.value).sum()

        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()
        # Calculate the count of values equal to the specified value for PySpark DataFrame
        k = df.filter(df[self.column] == self.value).count()

        return {"total": n, "count": k, "delta": k / n}

@dataclass
class CountBelowValue(Metric):
    """Number of values below threshold"""

    column: str
    value: float
    strict: bool = False

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = (df[self.column] < self.value).sum()

        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()
        # Calculate the count of values equal to the specified value for PySpark DataFrame
        k = df.filter(df[self.column] < self.value).count()

        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountBelowColumn(Metric):
    """Count how often column X below Y"""

    column_x: str
    column_y: str
    strict: bool = False

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        if self.strict:
            k = (df[self.column_x] < df[self.column_y]).sum()
        else:
            k = (df[self.column_x] <= df[self.column_y]).sum()

        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()
        df = df.filter(~df[self.column_x].isNull() & ~isnan(df[self.column_x]) & ~df[self.column_y].isNull() & ~isnan(df[self.column_y]))
        # Calculate the count of values satisfying the condition for PySpark DataFrame
        if self.strict:

            k = df.filter(df[self.column_x] < df[self.column_y]).count()
        else:
            k = df.filter(df[self.column_x] <= df[self.column_y]).count()

        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountRatioBelow(Metric):
    """Count how often X / Y below Z"""

    column_x: str
    column_y: str
    column_z: str
    strict: bool = False

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        if self.strict:
            k = (df[self.column_x] / df[self.column_y] < df[self.column_z]).sum()
        else:
            k = (df[self.column_x] / df[self.column_y] <= df[self.column_z]).sum()

        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()
        df = df.filter(~df[self.column_x].isNull() & ~isnan(df[self.column_x]) & ~df[self.column_y].isNull() & ~isnan(
            df[self.column_y]) & ~df[self.column_z].isNull() & ~isnan(df[self.column_z]))
        # Calculate the count of values satisfying the condition for PySpark DataFrame

        if self.strict:
            k = df.filter(df[self.column_x] / df[self.column_y] < df[self.column_z]).count()
        else:
            k = df.filter(df[self.column_x] / df[self.column_y] <= df[self.column_z]).count()

        return {"total": n, "count": k, "delta": k / n}



@dataclass
class CountCB(Metric):
    """Calculate lower/upper bounds for N%-confidence interval"""

    column: str
    conf: float = 0.95

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:

         # Define the confidence level (N%)
        confidence_level = (1 - self.conf) / 2
        # Calculate the lower and upper bounds for the confidence interval
        lcb = df[self.column].quantile(confidence_level)
        ucb = df[self.column].quantile(1 - confidence_level)

        return {"lcb": lcb, "ucb": ucb}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:

        # Calculate the confidence level (N%)
        confidence_level = (1 - self.conf) / 2
        # Calculate the lower and upper bounds for the confidence interval
        lcb = df.approxQuantile(self.column, [confidence_level], relativeError=0.0)[0]
        ucb = df.approxQuantile(self.column, [1 - confidence_level], relativeError=0.0)[0]

        return {"lcb": lcb, "ucb": ucb}

@dataclass
class CountLag(Metric):
    """A lag between latest date and today"""

    column: str
    fmt: str = "%Y-%m-%d"

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:

        df[self.column] = pd.to_datetime(df[self.column], format=self.fmt, errors="coerce")
        b = df[self.column].max()
        a = datetime.datetime.now()
        lag = (a - b).days if b is not pd.NaT else None

        return {"today": a.strftime(self.fmt), "last_day": b.strftime(self.fmt), "lag": lag}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:

        from pyspark.sql.functions import max
        from pyspark.sql.functions import current_date

        # Calculate the latest date (b) in PySpark DataFrame
        b_df = df.select(max(self.column).alias("max_date")).collect()
        b = b_df[0]["max_date"]
        b = datetime.datetime.strptime(b, self.fmt)

        # Calculate the current date (a)
        a = datetime.datetime.now()

        # Calculate the lag in days
        lag = (a - b).days if b is not None else None

        return {"today": a.strftime(self.fmt), "last_day": b.strftime(self.fmt) if b is not None else None, "lag": lag}