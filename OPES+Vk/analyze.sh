#!/bin/bash


output_file="COLVAR_m3s3"
> $output_file


first_file=true


for iter_dir in m3s3p*; do
    echo "Processing $iter_dir..."
    for opes_dir in "$iter_dir"; do
        opes_dir_name=$(basename "$opes_dir")

        if [ -f "$opes_dir/COLVAR" ]; then
            echo "Merging $opes_dir/COLVAR into $output_file..."

            if $first_file; then
                cat "$opes_dir/COLVAR" >> $output_file
                first_file=false
            else
                tail -n +2 "$opes_dir/COLVAR" >> $output_file
            fi
        else
            echo "No COLVARmod file found in $opes_dir"
        fi
    done
done

echo "All done!"
