{# Stock Sphinx autosummary class template, minus the ".. automethod:: __init__"
   line: conf.py's autodoc_default_options already sets special-members=__init__,
   so autoclass documents __init__ and the extra automethod made it a duplicate
   object description, which fails the Read the Docs build (fail_on_warning). #}
{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
