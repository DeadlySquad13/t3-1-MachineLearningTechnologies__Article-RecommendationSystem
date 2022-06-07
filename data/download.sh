#! /usr/bin/env bash

export YT_PROXY=Hahn
path="//home/geosearch/jsingh/ML_contest"

function download_table {
    table_name=$1
    output_name=$2
    columns=$3

    echo "Download <$table_name>..."
    format="<columns=[$columns];enable_column_names_header=true;missing_value_mode=print_sentinel;field_separator=\",\";enable_type_conversion=true>schemaful_dsv"
    yt download "$path/$table_name" --format="$format" > "$output_name.csv"
}


download_table  reviews_encoded          reviews         "user_id;org_id;rating;ts;aspects";
download_table  organisations_encoded    organisations   "org_id;city;average_bill;rating;rubrics_id;features_id";
download_table  users_encoded            users           "user_id;city";
download_table  aspects                  aspects         "aspect_id;aspect_name";
download_table  features                 features        "feature_id;feature_name";
download_table  rubrics                  rubrics         "rubric_id;rubric_name";
download_table  test_users_encoded       test_users      "user_id";
