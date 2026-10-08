# Step 1 - Exploratory analysis of DataCo

## Q1. What is in the data?

* Raw file: **180,519 rows x 53 columns** (one row per order line).
* Removed **7,754 cancelled lines** (never shipped) and 10 personal fields.
* Clean reference table: **172,765 rows x 14 columns**.
* Period: 2015-01-01 to 2018-01-31; 65,752 orders, 118 products, 11 departments.
* Missing values in kept columns: 0.
* Overall late rate: **57.3%**.

Category sizes:

* `shipping_mode`: Standard Class 60%, Second Class 20%, First Class 15%, Same Day 5%
* `market`: LATAM 29%, Europe 28%, Pacific Asia 23%, USCA 14%, Africa 6%
* `customer_segment`: Consumer 52%, Corporate 30%, Home Office 18%
* `department`: Fan Shop 37%, Apparel 27%, Golf 18%, Footwear 8%, Outdoors 5%, Fitness 1%, Discs Shop 1%, Technology 1%, Pet Shop 0%, Book Shop 0%, Health and Beauty 0%
* `payment_type`: DEBIT 40%, TRANSFER 24%, PAYMENT 24%, CASH 11%

## Q2. How are delivery promises set?

Rows = shipping mode, columns = scheduled days, cells = number of order lines.

| shipping_mode   |    0 |     1 |     2 |      4 |
|:----------------|-----:|------:|------:|-------:|
| Same Day        | 9293 |     0 |     0 |      0 |
| First Class     |    0 | 26513 |     0 |      0 |
| Second Class    |    0 |     0 | 33806 |      0 |
| Standard Class  |    0 |     0 |     0 | 103153 |

Every mode has exactly one promise: **scheduled days are a fixed function of shipping mode.**

## Q3. What do real delivery times look like?

Share of lines by real shipping days, per mode:

| shipping_mode   |     0 |     1 |     2 |     3 |     4 |     5 |     6 |
|:----------------|------:|------:|------:|------:|------:|------:|------:|
| Same Day        | 0.521 | 0.479 | 0     | 0     | 0     | 0     | 0     |
| First Class     | 0     | 0     | 1     | 0     | 0     | 0     | 0     |
| Second Class    | 0     | 0     | 0.202 | 0.2   | 0.2   | 0.2   | 0.198 |
| Standard Class  | 0     | 0     | 0.202 | 0.201 | 0.199 | 0.196 | 0.202 |

Late rate per mode:

| shipping_mode   |   late_rate |
|:----------------|------------:|
| Same Day        |       0.479 |
| First Class     |       1     |
| Second Class    |       0.798 |
| Standard Class  |       0.398 |

* `late == (days_real > days_scheduled)` holds for **100.0%** of rows: the late flag is a rule, not a judgement.
* Second and Standard Class: real days are spread **evenly over 2-6** (about 20% each).
* First Class: always 2 days against a promise of 1, so **100% late**.

## Q4. Does anything besides shipping mode explain delays?

Inside Standard Class only (so shipping mode is fixed), late rate per group of each variable.
A chi-square test checks whether the differences are bigger than chance would produce.

| factor           |   groups |   min_late |   max_late |   spread_pp |   chi2_p_value |
|:-----------------|---------:|-----------:|-----------:|------------:|---------------:|
| market           |        5 |      0.395 |      0.4   |       0.482 |          0.854 |
| customer_segment |        3 |      0.39  |      0.403 |       1.374 |          0.002 |
| department       |       11 |      0.367 |      0.414 |       4.663 |          0.739 |
| payment_type     |        4 |      0.388 |      0.402 |       1.332 |          0.085 |
| order_month      |       12 |      0.391 |      0.407 |       1.585 |          0.46  |
| quantity         |        5 |      0.393 |      0.4   |       0.759 |          0.809 |

* Chi-square test that Standard Class real days are uniform over 2-6: p = 0.01, but the shares only range from 19.6% to 20.2%.
* Late rates move by at most a few percentage points. With 100,000 rows, a test can flag differences that are real but tiny: customer segment is statistically significant, yet its whole effect is about 1.4 percentage points. **Statistically detectable is not the same as useful for prediction.**
* **Conclusion: once shipping mode is known, delays are almost pure noise.** DataCo looks like a synthetic dataset itself, with delivery days drawn close to uniformly.

## Q5. Hidden structures a generator must keep

**Markets come in time blocks.** Order lines per quarter and market:

| quarter   |   Africa |   Europe |   LATAM |   Pacific Asia |   USCA |
|:----------|---------:|---------:|--------:|---------------:|-------:|
| 2015Q1    |        0 |        0 |   14728 |              0 |      0 |
| 2015Q2    |        0 |     5024 |    9886 |              0 |      0 |
| 2015Q3    |        0 |    15075 |       0 |              0 |      0 |
| 2015Q4    |        0 |     3772 |       0 |          11469 |      0 |
| 2016Q1    |        0 |        0 |       0 |          14809 |      0 |
| 2016Q2    |        0 |        0 |       0 |            150 |  14618 |
| 2016Q3    |     2962 |      957 |       0 |           1891 |   9324 |
| 2016Q4    |     6978 |     2774 |       0 |           4884 |    579 |
| 2017Q1    |     1214 |      545 |   11986 |           1002 |    106 |
| 2017Q2    |        0 |     2036 |   12709 |              0 |      0 |
| 2017Q3    |        0 |    15094 |       0 |              0 |      0 |
| 2017Q4    |        0 |     2813 |       0 |           3343 |      0 |
| 2018Q1    |        0 |        0 |       0 |           2037 |      0 |

Only one or two markets are active in most quarters, so market depends strongly on date.

**Each department has a small fixed price list:**

| department        |   n_prices |    min |     max |
|:------------------|-----------:|-------:|--------:|
| Apparel           |          8 |  59.08 |  461.48 |
| Book Shop         |          1 |  31.08 |   31.08 |
| Discs Shop        |          4 |  11.29 |  260.65 |
| Fan Shop          |          9 |  11.54 |  399.98 |
| Fitness           |         14 |  22    |  999.99 |
| Footwear          |         13 |  27.99 | 1999.99 |
| Golf              |          6 |  25    |  199.99 |
| Health and Beauty |          1 | 293.04 |  293.04 |
| Outdoors          |         34 |   9.99 |  599.99 |
| Pet Shop          |          1 |  84.4  |   84.4  |
| Technology        |          3 | 252.88 | 1500    |

**Discounts take only 18 values** (0%, 1%, 2%, 3%, 4%, 5%, 6%, 7%, 9%, 10%, 12%, 13%, 15%, 16%, 17%, 18%, 20%, 25%).
 Quantity takes only 5 values (1-5).

## What this means for the generators

1. They must keep **deterministic rules**: promise = f(mode) and late = real > promise.
2. They must keep **conditional structures**: market given date, price given department.
3. Reproducing DataCo exactly gives data where delays are unpredictable. To be useful for procurement models, a generator should keep the overall delay rates but give delays **causes** (supplier reliability, peak season), and let us change them on purpose.
