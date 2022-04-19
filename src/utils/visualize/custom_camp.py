import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

__all__ = ["register_dbz_cmap", "register_rain_cmap"]


# standard color map for radar echo visualization
def register_dbz_cmap():
    dbz_list = ["#0000F6", "#01A0F6", "#00ECEC", "#01FF00", "#00C800", "#019000",
                "#FFFF00", "#E7C000", "#FF9000", "#FF0000", "#D60000", "#C00000",
                "#FF00F0", "#780084", "#AD90F0"]
    my_cmap = LinearSegmentedColormap.from_list('dbz', dbz_list, N=15)

    cm.register_cmap(cmap=my_cmap)


# standard color map for precipitation visualization
def register_rain_cmap():
    rain_list = ["#A6F28F", "#3DBA3D", "#61B8FF", "#0000E1", "#FA00FA"]

    cmap = mpl.colors.ListedColormap(rain_list)
    cmap.set_bad(alpha=0.0)
    cmap.set_over("#800040")
    cmap.set_under(alpha=0.0)

    bounds = [1, 1.5, 7, 15, 40, 50]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm
