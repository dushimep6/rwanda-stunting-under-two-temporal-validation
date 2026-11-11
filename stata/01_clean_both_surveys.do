*==============================================================================
* RWANDA DHS 2019-20 (RWKR81FL) AND 2024-25 (RWKR91FL): FULL CLEANING, THEN APPEND
* Uses the SAME cleaning rules as 01_data_cleaning_and_primary_regression.do
* (copied from it), once per survey. Saves three files for Python:
*   rwanda_stunting_clean_2020_v4.dta    2019-20 only
*   rwanda_stunting_clean_2025_b_v4.dta  2024-25 only (same rules; matches your v4 file)
*   rwanda_stunting_clean_pooled_v4.dta  both surveys appended, cluster_id unique
* Stata 17.
* Before running: set the three paths below to your own folders (DHS files from https://dhsprogram.com).
* Keep every output on your own computer. Never upload .dta, .csv or .log files.
*==============================================================================
version 17.0
clear all
set more off
capture log close _all

local dir    "PATH\TO\YOUR\OUTPUT\FOLDER"
local f2020  "PATH\TO\RWKR81FL.DTA"      // 2019-20 children's recode
local f2025  "PATH\TO\RWKR91FL.dta"      // 2024-25 children's recode
* No log file is written on purpose: Stata logs contain the licence number and must not be uploaded.

capture program drop clean_full
program define clean_full
    args yr
    foreach v in hw13 hw1 hw70 v005 v021 v022 b4 bord b0 v106 v155 ///
        v190 v025 v024 v113 v115 v116 v160 v481 v467d m4 m5 m19 m18 ///
        v445 v438 m14 m13 m45 m60 m15 v012 v136 v137 v404 m17 {
        capture confirm variable `v'
        if _rc {
            display as error "Required source variable `v' is absent in survey `yr'."
            exit 111
        }
    }
keep if hw13 == 0
keep if inrange(hw1, 0, 59)

*------------------------------------------------------------------------------
* 3. OUTCOME: STUNTING
*------------------------------------------------------------------------------
* DHS HAZ (HW70) is multiplied by 100. Stunting is HAZ strictly below -2 SD.
gen byte stunted = (hw70 < -200) if inrange(hw1, 0, 59) & inrange(hw70, -600, 600)
label define stunt_lbl 0 "Not stunted" 1 "Stunted", replace
label values stunted stunt_lbl
label variable stunted "Child stunting (HAZ < -2 SD)"
drop if missing(stunted)
assert !missing(stunted)
assert inrange(hw1, 0, 59)
assert inrange(hw70, -600, 600)

*------------------------------------------------------------------------------
* 4. DERIVE AND CLEAN PREDICTORS
*------------------------------------------------------------------------------
* Child characteristics
gen byte first_birth = (bord == 1) if !missing(bord)
label variable first_birth "First-born child"

gen birth_weight = m19 if !missing(m19) & m19 < 9990
label variable birth_weight "Birth weight (g)"
gen byte lbw = (birth_weight < 2500) if !missing(birth_weight)
label define lbw_lbl 0 "Not low birth weight" 1 "Low birth weight", replace
label values lbw lbw_lbl

gen byte birth_size = m18 if inrange(m18, 1, 5)
label variable birth_size "Reported size at birth"

* Breastfeeding: M4 contains the status codes; M5 contains months.
* M4 codes 93/94/95/96 indicate not currently/never/still/until died.
* M4 97/98 are inconsistent/don't know and are left missing.
gen bf_months = m5 if inrange(m5, 0, 59)
label variable bf_months "Months of breastfeeding"
gen byte still_breastfeeding = .
replace still_breastfeeding = 1 if m4 == 95
replace still_breastfeeding = 0 if inrange(m4, 0, 59) | inlist(m4, 93, 94, 96)
label define bf_lbl 0 "Not currently breastfeeding" 1 "Currently breastfeeding", replace
label values still_breastfeeding bf_lbl
label variable still_breastfeeding "Currently breastfeeding"

* Maternal anthropometry. V445 is BMI x100; V438 is height x10.
gen double mother_bmi = v445/100 if inrange(v445, 1000, 6000)
label variable mother_bmi "Maternal BMI (kg/m2)"
gen byte bmi_missing = missing(mother_bmi)
label variable bmi_missing "Maternal BMI missing"
gen mother_height = v438/10 if inrange(v438, 1000, 2200)
label variable mother_height "Maternal height (cm)"
gen byte literacy = v155 if inrange(v155, 0, 2)
label variable literacy "Maternal literacy category"

* V457 was found entirely missing in the prior run; do not use as a feature.
capture drop v457

* Antenatal/delivery variables retained for screening and downstream ML.
gen anc_visits = m14 if inrange(m14, 0, 20)
label variable anc_visits "Number of antenatal care visits"
gen anc_timing = m13 if inrange(m13, 0, 9)
label variable anc_timing "Timing of first antenatal care visit"
gen byte iron_supp = (m45 == 1) if inrange(m45, 0, 1)
label define yesno_lbl 0 "No" 1 "Yes", replace
label values iron_supp yesno_lbl
label variable iron_supp "Received iron supplementation"
gen byte deworming = (m60 == 1) if inrange(m60, 0, 1)
label values deworming yesno_lbl
label variable deworming "Received deworming medication"

* Delivery place categories. Codes were checked against observed M15 labels.
gen byte delivery_place_grp = .
replace delivery_place_grp = 1 if inlist(m15, 11, 12)
replace delivery_place_grp = 2 if inlist(m15, 21, 22, 23, 24)
replace delivery_place_grp = 3 if inlist(m15, 31, 32)
replace delivery_place_grp = 4 if m15 == 96
label define delplace_lbl 1 "Home" 2 "Public facility" 3 "Private facility" 4 "Other", replace
label values delivery_place_grp delplace_lbl
label variable delivery_place_grp "Place of delivery"

gen byte delivery_facility = .
replace delivery_facility = 1 if inlist(delivery_place_grp, 2, 3)
replace delivery_facility = 0 if inlist(delivery_place_grp, 1, 4)
label define delfac_lbl 0 "Home/other" 1 "Facility", replace
label values delivery_facility delfac_lbl
label variable delivery_facility "Delivery in health facility"

* WASH: V115=996 means source on premises; code as 0 minutes for this model.
gen water_time = v115
replace water_time = 0 if v115 == 996
replace water_time = . if inlist(v115, 997, 998)
label variable water_time "Time to obtain drinking water (minutes)"

gen byte shared_toilet = v160 if inrange(v160, 0, 1)
label define shared_lbl 0 "Not shared" 1 "Shared", replace
label values shared_toilet shared_lbl
label variable shared_toilet "Household toilet shared"

gen byte distance_problem = (v467d == 1) if inlist(v467d, 1, 2)
label define distance_lbl 0 "Not a big problem" 1 "Big problem", replace
label values distance_problem distance_lbl
label variable distance_problem "Distance to health facility is a big problem"

* Age groups to avoid assuming a linear log-odds change for every month of age.
gen byte age_group = .
replace age_group = 1 if inrange(hw1, 0, 5)
replace age_group = 2 if inrange(hw1, 6, 11)
replace age_group = 3 if inrange(hw1, 12, 23)
replace age_group = 4 if inrange(hw1, 24, 35)
replace age_group = 5 if inrange(hw1, 36, 47)
replace age_group = 6 if inrange(hw1, 48, 59)
label define age_lbl 1 "0-5 months" 2 "6-11 months" 3 "12-23 months" ///
    4 "24-35 months" 5 "36-47 months" 6 "48-59 months", replace
label values age_group age_lbl
label variable age_group "Child age group"

*------------------------------------------------------------------------------
* 5. COLLAPSE WATER SOURCE / TOILET CATEGORIES
*------------------------------------------------------------------------------
* Improved source grouping follows current WHO/UNICEF JMP source-type framing.
* Delivered and bottled water are grouped as improved source types; this does
* not itself mean the water service is safely managed.
gen byte water_source_grp = .
replace water_source_grp = 1 if inlist(v113, 11, 12, 13, 14, 21, 31, 41, 51, 61)
replace water_source_grp = 1 if inlist(v113, 62, 71)
replace water_source_grp = 2 if inlist(v113, 32, 42, 43)
label define watersrc_lbl 1 "Improved" 2 "Unimproved", replace
label values water_source_grp watersrc_lbl
label variable water_source_grp "Drinking water source type"

gen byte toilet_grp = .
replace toilet_grp = 1 if inlist(v116, 11, 12, 13, 21, 22, 41)
replace toilet_grp = 2 if v116 == 23
replace toilet_grp = 3 if v116 == 31
* 2019-20 only: flush to somewhere else (14) or unknown place (15) = unimproved (JMP); none in 2025.
replace toilet_grp = 2 if inlist(v116, 14, 15)
label define toilet_lbl 1 "Improved" 2 "Unimproved" 3 "No facility", replace
label values toilet_grp toilet_lbl
label variable toilet_grp "Sanitation facility"

* Code 96 means "not a de jure resident" in this file and must remain missing.
assert missing(water_source_grp) if v113 == 96
assert missing(toilet_grp) if v116 == 96

* V160 is structurally missing when no toilet exists; no toilet cannot be shared.
replace shared_toilet = 0 if toilet_grp == 3 & missing(shared_toilet)


gen double wt = v005/1000000
rename v021 cluster_id
rename v022 strata_id
    gen int survey = `yr'
end

*------------------------------------------------------------------------------
* CLEAN EACH SURVEY AND SAVE IT SEPARATELY
*------------------------------------------------------------------------------
use "`f2020'", clear
clean_full 2020
count
tab stunted
save "`dir'\rwanda_stunting_clean_2020_v4.dta", replace

use "`f2025'", clear
clean_full 2025
count
tab stunted
save "`dir'\rwanda_stunting_clean_2025_b_v4.dta", replace

*------------------------------------------------------------------------------
* APPEND AND SAVE THE POOLED FILE (cluster and strata made unique)
*------------------------------------------------------------------------------
use "`dir'\rwanda_stunting_clean_2025_b_v4.dta", clear
append using "`dir'\rwanda_stunting_clean_2020_v4.dta"

replace cluster_id = survey*10000 + cluster_id
replace strata_id  = survey*1000  + strata_id

* weights: rescale within each survey so each survey counts for its own sample size
bysort survey: egen double sumw = total(wt)
bysort survey: gen double nsurv = _N
replace wt = wt * nsurv / sumw
drop sumw nsurv
svyset cluster_id [pweight=wt], strata(strata_id)

count
tab survey stunted
svy: proportion stunted, over(survey)
save "`dir'\rwanda_stunting_clean_pooled_v4.dta", replace

display as text "===== Done. Expect 2020 = 3,809, 2025 = 6,715, pooled = 10,524 ====="
