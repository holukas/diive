#!/usr/bin/env python
"""
Simple script to generate static HTML documentation site
"""
import os
import shutil
from pathlib import Path

# Create output directory
output_dir = Path("docs/source/_build/html")
output_dir.mkdir(parents=True, exist_ok=True)

# Create index.html
index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>diive Documentation</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        .section { background: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        ul { line-height: 1.8; }
        a { color: #3498db; text-decoration: none; }
        a:hover { text-decoration: underline; }
        code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-family: 'Courier New', monospace; }
    </style>
</head>
<body>
    <h1>📚 diive Documentation</h1>
    <p><strong>Time series processing library for ecosystem observations</strong></p>

    <div class="section">
        <h2>Getting Started</h2>
        <ul>
            <li><a href="https://github.com/holukas/diive">GitHub Repository</a></li>
            <li><a href="https://github.com/holukas/diive/blob/main/README.md">README</a></li>
            <li><a href="https://github.com/holukas/diive/blob/main/CHANGELOG.md">Changelog</a></li>
        </ul>
    </div>

    <div class="section">
        <h2>Documentation Pages</h2>
        <p>Documentation is currently hosted on GitHub. Please visit:</p>
        <ul>
            <li><a href="https://github.com/holukas/diive/tree/main/docs/source">Documentation Source Files</a></li>
            <li><a href="https://github.com/holukas/diive/tree/main/notebooks">Example Notebooks</a></li>
        </ul>
    </div>

    <div class="section">
        <h2>Example Notebooks</h2>
        <p>Browse examples organized by topic:</p>
        <ul>
            <li><a href="https://github.com/holukas/diive/tree/main/notebooks/io">Data I/O</a></li>
            <li><a href="https://github.com/holukas/diive/tree/main/notebooks/timeseries">Time Series</a></li>
            <li><a href="https://github.com/holukas/diive/tree/main/notebooks/variables">Variable Creation</a></li>
            <li><a href="https://github.com/holukas/diive/tree/main/notebooks/qc">Quality Control</a></li>
            <li><a href="https://github.com/holukas/diive/tree/main/notebooks/analyses">Analysis</a></li>
            <li><a href="https://github.com/holukas/diive/tree/main/notebooks/gapfilling">Gap-Filling</a></li>
            <li><a href="https://github.com/holukas/diive/tree/main/notebooks/flux">Flux Processing</a></li>
            <li><a href="https://github.com/holukas/diive/tree/main/notebooks/plotting">Visualization</a></li>
        </ul>
    </div>

    <div class="section">
        <h2>Features</h2>
        <ul>
            <li>📥 Load and read EddyPro, TOA5, and custom data formats</li>
            <li>🔧 Quality control and outlier detection</li>
            <li>⬜ Gap-filling (MDS, Random Forest, XGBoost)</li>
            <li>📊 Statistical analysis and aggregation</li>
            <li>📈 Publication-quality visualization</li>
            <li>💨 Complete flux processing workflows</li>
        </ul>
    </div>

    <div class="section">
        <p style="text-align: center; color: #888; margin-top: 40px;">
            <small>diive v0.91.0 | <a href="https://www.swissfluxnet.ethz.ch/">Swiss FluxNet</a> | <a href="https://grassland.ethz.ch/">ETH Grassland Sciences</a></small>
        </p>
    </div>
</body>
</html>
"""

# Write index.html
(output_dir / "index.html").write_text(index_html)
print(f"✅ Created documentation at {output_dir / 'index.html'}")
print(f"✅ Documentation site ready for deployment!")
