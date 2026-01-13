from diive.pkgs.createvar.daynightflag import DaytimeNighttimeFlag


def create_daytime_nighttime_flags(timestamp_index, lat, lon, utc_offset):
    # Detect daytime and nighttime
    dnf = DaytimeNighttimeFlag(
        timestamp_index=timestamp_index,
        nighttime_threshold=20,
        lat=lat, lon=lon,
        utc_offset=utc_offset)
    flag_daytime = dnf.get_daytime_flag()
    flag_nighttime = dnf.get_nighttime_flag()  # 0/1 flag needed outside init

    is_daytime = flag_daytime == 1  # Convert 0/1 flag to False/True flag
    is_nighttime = flag_nighttime == 1  # Convert 0/1 flag to False/True flag
    return flag_daytime, flag_nighttime, is_daytime, is_nighttime
