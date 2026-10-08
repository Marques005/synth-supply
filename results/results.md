# Benchmark results

All metrics compare against the real **holdout** (30% of DataCo, never seen by the generators).
The *Real train* row scores the real training data the same way: the best any generator could do.

## Summary

| generator              |   column_shapes |   column_pairs |   rule_validity |   TSTR_AUC |   exact_copy_rate |   median_DCR |
|:-----------------------|----------------:|---------------:|----------------:|-----------:|------------------:|-------------:|
| Real train (reference) |           0.998 |          0.99  |           1     |      0.751 |             0.02  |        0.85  |
| Independent            |           0.997 |          0.943 |           0.204 |      0.493 |             0     |        1.726 |
| Gaussian copula        |           0.997 |          0.954 |           0.409 |      0.709 |             0.001 |        1.506 |
| Domain simulator       |           0.996 |          0.973 |           1     |      0.733 |             0.015 |        0.96  |

## Late rate by shipping mode

| shipping_mode   |   Real holdout |   Independent |   Gaussian copula |   Domain simulator |
|:----------------|---------------:|--------------:|------------------:|-------------------:|
| Same Day        |          0.477 |         0.563 |             0.757 |              0.481 |
| First Class     |          1     |         0.577 |             0.689 |              1     |
| Second Class    |          0.799 |         0.571 |             0.629 |              0.771 |
| Standard Class  |          0.396 |         0.577 |             0.509 |              0.414 |

## The copula's five weakest column pairs

|                                |   Independent |   Gaussian copula |   Domain simulator |
|:-------------------------------|--------------:|------------------:|-------------------:|
| department x unit_price        |         0.372 |             0.427 |              0.989 |
| market x order_month           |         0.493 |             0.546 |              0.992 |
| shipping_mode x days_scheduled |         0.422 |             0.623 |              0.996 |
| days_real x late               |         0.707 |             0.771 |              0.963 |
| shipping_mode x days_real      |         0.714 |             0.786 |              0.963 |
