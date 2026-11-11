# Stata cleaning step

`01_clean_both_surveys.do` (Stata 17) cleans the two DHS children's recode files with the same rules and saves the files that the Python scripts read:

| Output | Content |
|---|---|
| `rwanda_stunting_clean_2020_v4.dta` | 2019-20 only (expected 3,809 children) |
| `rwanda_stunting_clean_2025_b_v4.dta` | 2024-25 only (expected 6,715 children) |
| `rwanda_stunting_clean_pooled_v4.dta` | Both surveys appended, with unique cluster and strata codes (10,524 children) |

Inputs are the 2019-20 file `RWKR81FL.DTA` and the 2024-25 file `RWKR91FL.dta`, from the DHS Program (https://dhsprogram.com).

Before running, edit the three paths at the top of the file. Keep the `.dta` outputs on your own computer. Do not write or upload a Stata log, because it contains the licence number.
