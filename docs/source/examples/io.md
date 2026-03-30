# Loading & Reading Data

Learn how to load data from various file formats into diive.

## Key Modules

- `diive.core.io.ReadFileType` — Auto-detect and read file format
- `diive.core.io.DataFileReader` — Read single files with format specification
- `diive.core.io.MultiDataFileReader` — Read and concatenate multiple files

## Example Notebooks

Detailed working examples are in the notebooks:

- [Read single EddyPro file with DataFileReader](../../notebooks/io/Read_single_EddyPro_fluxnet_output_file_with_DataFileReader.ipynb)
- [Read single EddyPro file with ReadFileType](../../notebooks/io/Read_single_EddyPro_fluxnet_output_file_with_ReadFileType.ipynb)
- [Read multiple EddyPro files](../../notebooks/io/Read_multiple_EddyPro_fluxnet_output_files_with_MultiDataFileReader.ipynb)
- [Load/Save Parquet files](../../notebooks/io/LoadSaveParquetFile.ipynb)
- [Format EddyPro for FLUXNET upload](../../notebooks/io/FormatEddyProFluxnetFileForUpload.ipynb)
- [Format meteorological data for EddyPro](../../notebooks/io/FormatMeteoForEddyProFluxProcessing.ipynb)
- [Format meteorological data for FLUXNET](../../notebooks/io/FormatMeteoForFluxnetUpload.ipynb)

## Supported Formats

- **EddyPro output**: Full Flux output and biomet files
- **TOA5**: Campbell Scientific datalogger format
- **Parquet**: Columnar binary format (recommended for large files)
- **CSV**: Comma-separated values
- **Custom formats**: Implement your own `DataFileReader`

## Quick Example

```python
import diive as dv

# Auto-detect format
reader = dv.ReadFileType(filepath='data/CH-Oe2_EddyPro_output.txt')
df = reader.read()

# Or specify format explicitly
from diive.core.io import DataFileReader
reader = DataFileReader(file_path='data/file.txt', file_type='EddyPro')
df = reader.read()

# Save processed data
dv.save_parquet(df, 'processed_data.parquet')
```

## See Also

- [FAQ: Data Loading](../faq.md#data-loading--io)
- [System Architecture](../guide/architecture.md)
