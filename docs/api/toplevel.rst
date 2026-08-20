.. _api_toplevel:

Top level (``dv.*``)
====================

File I/O, dataframe helpers and verbosity control, reached directly on the
package rather than through a namespace.

.. currentmodule:: diive

.. autosummary::
   :toctree: ../_autosummary

   load_exampledata_parquet
   load_exampledata_parquet_lae
   load_parquet
   load_parquet_many
   save_parquet
   to_diive_format
   ReadFileType
   search_files
   keep_vars
   keep_records_where
   transform_yearmonth_matrix_to_longform
   sstats
   get_encoded_value_from_int
   get_encoded_value_series
   set_verbosity
   get_verbosity
