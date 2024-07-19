# nc_file_validation_tool
This tool compares files generated out of the Generate process with what OBPG provides to confirm they're similar

## Operational Instructions
1. `poetry install` to install dependencies
2. Have `.nc` files placed in `nc_files` folder
3. Create `nc_files.csv` with mapping like follows
```
processed1.nc,obpg_sst_1.nc,obpg_sst4_1.nc
processed2.nc,obpg_sst_2.nc,obpg_oc_2.nc
processed3.nc,obpg_sst_3.nc,obpg_sst4_3.nc
processed4.nc,obpg_sst_4.nc,obpg_oc_4.nc
```
4. simply running the program like `poetry run python nc_file_validation_tool/main.py` from the project would start it and generate a json output