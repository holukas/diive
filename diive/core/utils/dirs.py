def create_default_outdir(ctx, run_id: str = 'DIIVE-YYYYMMDD-hhmmss'):
    """Create default folder name and location of output directory"""
    rundir = 'OUT_{}'.format(run_id)
    scriptdir = ctx.dir_script.parent  # Location dir of this script
    outdir = scriptdir / 'OUTPUT' / rundir  # Default output dir, collects run outputs
    return rundir, outdir
