"""DQ Report."""

from typing import Dict, List, Tuple, Union
from dataclasses import dataclass
from user_input.metrics import Metric

import pandas as pd
import pyspark.sql as ps

LimitType = Dict[str, Tuple[float, float]]
CheckType = Tuple[str, Metric, LimitType]


@dataclass
class Report:
    """DQ report class."""

    checklist: List[CheckType]
    engine: str = "pandas"

    def fit(self, tables: Dict[str, Union[pd.DataFrame, ps.DataFrame]]) -> Dict:
        """Calculate DQ metrics and build report."""

        if self.engine == "pandas":
            return self._fit_pandas(tables)

        if self.engine == "pyspark":
            return self._fit_pyspark(tables)

        raise NotImplementedError("Only pandas and pyspark APIs currently supported!")

    def _fit_pandas(self, tables: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate DQ metrics and build report.  Engine: Pandas"""
        self.report_ = {}
        report = self.report_

        report['title'] = f'DQ Report for tables {sorted(list(set(tables.keys())))}'

        report['result'] = pd.DataFrame()
        report['passed'] = 0
        report['failed'] = 0
        report['errors'] = 0
        report['total'] = 0
        table_report = {
            'table_name': [],
            'metric': [],
            'limits': [],
            'values': [],
            'status': [],
            'error': []
        }
        for table_n, metric, limits in self.checklist:
            try:
                table = tables[table_n]
                result = metric(table)
                # Check if the metric values are within specified limits
                status = '.'
                for key, (lower_limit, upper_limit) in limits.items():
                    value = result.get(key, None)
                    if value is not None and (lower_limit <= value <= upper_limit):
                        continue
                    else:
                        status = 'F'
                        break

                # Add the metric results to the table report
                table_report['table_name'].append(table_n)
                table_report['metric'].append(str(metric))
                table_report['limits'].append(str(limits))
                table_report['values'].append(result)
                table_report['status'].append(status)
                table_report['error'].append('')

                # Update the overall report counters
                if status == '.':
                    report['passed'] += 1
                else:
                    report['failed'] += 1

            except Exception as e:
                # Handle any exceptions that occur during metric calculation
                status = 'E'
                table_report['table_name'].append(table_n)
                table_report['metric'].append(str(metric))
                table_report['limits'].append(str(limits))
                table_report['values'].append(result)
                table_report['status'].append(status)
                table_report['error'].append(str(e))
                report['errors'] += 1

            finally:
                report['total'] += 1

        report['result'] = pd.DataFrame.from_dict(table_report)
        suma = report['passed'] + report['failed'] + report['errors']
        report['passed_pct'] = round(report['passed'] * 100 / suma, 2)
        report['failed_pct'] = round(report['failed'] * 100 / suma, 2)
        report['errors_pct'] = round(report['errors'] * 100 / suma, 2)

        return report

    def _fit_pyspark(self, tables: Dict[str, ps.DataFrame]) -> Dict:
        """Calculate DQ metrics and build report.  Engine: PySpark"""
        self.report_ = {}
        report = self.report_

        report['title'] = f'DQ Report for tables {sorted(list(set(tables.keys())))}'

        report['result'] = pd.DataFrame()
        report['passed'] = 0
        report['failed'] = 0
        report['errors'] = 0
        report['total'] = 0
        table_report = {
            'table_name': [],
            'metric': [],
            'limits': [],
            'values': [],
            'status': [],
            'error': []
        }

        for table_n, metric, limits in self.checklist:
            # Calculate the metric and get the result
            try:
                table = tables[table_n]
                result = metric(table)

                # Check if the metric values are within specified limits
                status = '.'
                for key, (lower_limit, upper_limit) in limits.items():
                    value = result.get(key, None)
                    if value is not None and (lower_limit <= value <= upper_limit):
                        continue
                    else:
                        status = 'F'
                        break

                # Add the metric results to the table report
                table_report['table_name'].append(table_n)
                table_report['metric'].append(str(metric))
                table_report['limits'].append(str(limits))
                table_report['values'].append(result)
                table_report['status'].append(status)
                table_report['error'].append('')

                # Update the overall report counters
                if status == '.':
                    report['passed'] += 1
                else:
                    report['failed'] += 1

            except Exception as e:
            # Handle any exceptions that occur during metric calculation
                status = 'E'
                table_report['table_name'].append(table_n)
                table_report['metric'].append(str(metric))
                table_report['limits'].append(str(limits))
                table_report['values'].append(result)
                table_report['status'].append(status)
                table_report['error'].append(str(e))
                report['errors'] += 1

            finally:
                report['total'] += 1

        report['result'] = pd.DataFrame.from_dict(table_report)
        suma = report['passed'] + report['failed'] + report['errors']
        report['passed_pct'] = round(report['passed'] * 100 / suma, 2)
        report['failed_pct'] = round(report['failed'] * 100 / suma, 2)
        report['errors_pct'] = round(report['errors'] * 100 / suma, 2)

        return report


    def to_str(self) -> None:
        """Convert report to string format."""
        report = self.report_

        msg = (
            "This Report instance is not fitted yet. "
            "Call 'fit' before usong this method."
        )

        assert isinstance(report, dict), msg

        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)
        pd.set_option("display.max_colwidth", 20)
        pd.set_option("display.width", 1000)

        return (
            f"{report['title']}\n\n"
            f"{report['result']}\n\n"
            f"Passed: {report['passed']} ({report['passed_pct']}%)\n"
            f"Failed: {report['failed']} ({report['failed_pct']}%)\n"
            f"Errors: {report['errors']} ({report['errors_pct']}%)\n"
            "\n"
            f"Total: {report['total']}"
        )
