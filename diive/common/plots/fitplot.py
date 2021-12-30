def fitplot(label, ax, x, y, fitx, fity,
            fity_ci95_lower, fity_ci95_upper, fity_pb_lower, fity_pb_upper,
            color):


    # x/y
    line_xy = ax.scatter(x, y,
                         edgecolor='none', color=color, alpha=1, s=60,
                         label=label, zorder=99, marker='o')

    # Fit
    label_fit = r"$y = ax^2 + bx + c$"
    line_fit, = ax.plot(fitx, fity,
                       c=color, lw=2, zorder=98, alpha=1, label=label_fit)

    # Fit confidence region
    # Uncertainty lines (95% confidence)
    line_fit_ci = ax.fill_between(fitx, fity_ci95_lower, fity_ci95_upper,
                                  alpha=.2, color=color,zorder=1,label="95% confidence region")

    # Fit prediction interval
    # Lower prediction band (95% confidence)
    ax.plot(fitx, fity_pb_lower,
            color=color, ls='--', zorder=98,
            label="95% prediction interval")
    # Upper prediction band (95% confidence)
    line_fit_pb, = ax.plot(fitx, fity_pb_upper,
                          color=color, ls='--', zorder=98,
                          label="95% prediction interval")

    return line_xy, line_fit, line_fit_ci, line_fit_pb
