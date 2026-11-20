# Ridership model results (2026-11-20)

424 subway station complexes, target log(1 + mean weekday entries, 2025). Hexagon features: 127 (the station's 500 m hexagon); walk features: 122 (what is within an 800 m walk, `_walk` models). Scores are on stations the model did not train on. Final model (lowest held-out error among the Random Forests): **rf**.

## Whole neighborhoods held out (5 folds by NTA, mean of 3 shuffles; r2_log_sd = spread)

| model       |   r2_log |   mae_log |   rmse_log |   mae_riders |   rmse_riders |   median_abs_pct_err |   spearman |   r2_log_sd |
|:------------|---------:|----------:|-----------:|-------------:|--------------:|---------------------:|-----------:|------------:|
| mean        |   -0.015 |     0.767 |      0.985 |      6592.1  |       14257.2 |               58.444 |     -0.135 |       0.008 |
| people      |    0.378 |     0.6   |      0.771 |      5420.75 |       11965.1 |               47.162 |      0.606 |       0.014 |
| linear      |    0.57  |     0.487 |      0.64  |      4854.37 |       11378.8 |               37.522 |      0.759 |       0.017 |
| rf_no_stud  |    0.569 |     0.49  |      0.642 |      4882.65 |       11779.1 |               36.207 |      0.744 |       0.01  |
| rf          |    0.573 |     0.489 |      0.639 |      4856.88 |       11736.9 |               35.258 |      0.746 |       0.007 |
| people_walk |    0.439 |     0.577 |      0.732 |      5292.39 |       11673.5 |               46.603 |      0.632 |       0.011 |
| linear_walk |    0.595 |     0.484 |      0.621 |      4885.15 |       11550.4 |               38.281 |      0.764 |       0.012 |
| rf_walk     |    0.546 |     0.507 |      0.658 |      4980.19 |       11721.4 |               38.759 |      0.733 |       0.011 |
| rf_both     |    0.562 |     0.499 |      0.647 |      4929.58 |       11633.5 |               37.777 |      0.742 |       0.012 |

## Whole boroughs held out

| model       |   r2_log |   mae_log |   rmse_log |   mae_riders |   rmse_riders |   median_abs_pct_err |   spearman |   r2_log_sd |
|:------------|---------:|----------:|-----------:|-------------:|--------------:|---------------------:|-----------:|------------:|
| mean        |   -0.225 |     0.852 |      1.081 |      7047.74 |       14633.6 |               65.695 |     -0.408 |         nan |
| people      |    0.363 |     0.612 |      0.78  |      5497.65 |       12417.8 |               48.655 |      0.59  |         nan |
| linear      |    0.394 |     0.587 |      0.76  |      5531.31 |       10749.6 |               44.294 |      0.669 |         nan |
| rf_no_stud  |    0.472 |     0.533 |      0.71  |      5107.9  |       12634.3 |               37.165 |      0.714 |         nan |
| rf          |    0.469 |     0.535 |      0.712 |      5098.62 |       12574.9 |               37.787 |      0.706 |         nan |
| people_walk |    0.434 |     0.583 |      0.735 |      5322.68 |       12003.7 |               47.937 |      0.622 |         nan |
| linear_walk |    0.508 |     0.545 |      0.685 |      5020.48 |       11557.5 |               40.967 |      0.709 |         nan |
| rf_walk     |    0.442 |     0.556 |      0.73  |      5292.75 |       12659.9 |               39.712 |      0.683 |         nan |
| rf_both     |    0.454 |     0.548 |      0.722 |      5249.85 |       12655.5 |               38.669 |      0.696 |         nan |

`r2_log`, `mae_log`, `rmse_log` are on the log scale the model trains on; `mae_riders` is in weekday entries; `median_abs_pct_err` is the typical miss in percent; `spearman` is rank agreement (what matters for ranking sites).

## Most important features (Random Forest, permutation on held-out neighborhoods)

| feature                           |   rmse_increase_log |
|:----------------------------------|--------------------:|
| retail_floor_m2                   |               0.032 |
| nearest_other_station_m           |               0.021 |
| share_commute_drive_alone         |               0.02  |
| floor_area_m2                     |               0.019 |
| com_floor_m2                      |               0.012 |
| km_to_midtown                     |               0.01  |
| job_density                       |               0.007 |
| nearest_college_m                 |               0.006 |
| share_under18                     |               0.006 |
| built_far                         |               0.005 |
| nearest_other_rail_peak_per_hour  |               0.005 |
| jobs                              |               0.005 |
| bus_routes                        |               0.004 |
| bus_trips_per_km2                 |               0.004 |
| nearest_other_rail_weekday_trains |               0.003 |

## Second Avenue Subway check (trained without these three stations)

| station_complex   |   weekday_riders |   predicted |
|:------------------|-----------------:|------------:|
| 72 St (Q)         |            24817 |       17878 |
| 86 St (Q)         |            19537 |       26259 |
| 96 St (Q)         |            14176 |       14264 |

## Top 15 candidate hexagons by new riders

Eligible = not a park/airport/cemetery neighborhood, at most half unbuildable land, not Staten Island (no training stations there). New riders = predicted riders x share of the 800 m walk area not already within 800 m of the subway, LIRR, Metro-North or PATH.

| hex_id   | borough   | ntaname                              |   predicted_riders |   predicted_p10 |   predicted_p90 |   uncovered_share_any_rail |   new_riders |   near_any_proposal |
|:---------|:----------|:-------------------------------------|-------------------:|----------------:|----------------:|---------------------------:|-------------:|--------------------:|
| r041c025 | Manhattan | Hell's Kitchen                       |               9593 |            2937 |           25454 |                      0.865 |         8298 |                   0 |
| r056c042 | Bronx     | Co-op City                           |               8235 |            2553 |           22118 |                      1     |         8235 |                   0 |
| r018c033 | Brooklyn  | Marine Park-Mill Basin-Bergen Beach  |               6974 |            1959 |           22950 |                      1     |         6974 |                   1 |
| r042c026 | Manhattan | Upper West Side-Lincoln Square       |              13883 |            5996 |           30591 |                      0.495 |         6872 |                   0 |
| r036c028 | Manhattan | Stuyvesant Town-Peter Cooper Village |               7386 |            3062 |           23179 |                      0.905 |         6684 |                   0 |
| r037c027 | Manhattan | Murray Hill-Kips Bay                 |              17392 |            9013 |           38038 |                      0.354 |         6157 |                   0 |
| r037c039 | Queens    | Corona                               |               5664 |            2681 |           15304 |                      0.997 |         5647 |                   0 |
| r041c031 | Queens    | Queensbridge-Ravenswood-Dutch Kills  |               6393 |            3083 |           12712 |                      0.854 |         5460 |                   0 |
| r057c041 | Bronx     | Co-op City                           |               5367 |            2134 |           14892 |                      0.966 |         5185 |                   0 |
| r039c042 | Queens    | Flushing-Willets Point               |               5733 |            2567 |           14795 |                      0.87  |         4988 |                   0 |
| r033c027 | Manhattan | Lower East Side                      |               5543 |            2286 |           13359 |                      0.873 |         4839 |                   0 |
| r042c032 | Queens    | Old Astoria-Hallets Point            |               6064 |            2908 |           12859 |                      0.765 |         4639 |                   0 |
| r034c044 | Queens    | Pomonok-Electchester-Hillcrest       |               4953 |            2075 |           13950 |                      0.931 |         4611 |                   0 |
| r051c034 | Bronx     | Claremont Village-Claremont (East)   |               5063 |            2623 |           14185 |                      0.902 |         4567 |                   0 |
| r024c032 | Brooklyn  | East Flatbush-Rugby                  |               4624 |            2163 |           10501 |                      0.983 |         4545 |                   1 |
