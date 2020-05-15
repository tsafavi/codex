declare -a arr=("entities" "types")
for outer in "${arr[@]}"; do
    for inner in $(find ${outer} -maxdepth 1 -mindepth 1 -type d); do
        unzip ${inner}/extracts.zip -d ${inner}/
    done
done
