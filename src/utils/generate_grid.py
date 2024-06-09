import ee

def generate_grid(region, scale, pixelWidth=375, pixelHeight=375):
    bounds = region.geometry().bounds()
    coords = ee.List(bounds.coordinates().get(0))

    # extract the coordinates from AOI
    xmin = ee.List(coords.get(0)).get(0)
    ymin = ee.List(coords.get(0)).get(1)
    xmax = ee.List(coords.get(2)).get(0)
    ymax = ee.List(coords.get(2)).get(1)

    # calculate the width and height in meters of each tile
    width_in_meters = ee.Number(pixelWidth).multiply(scale)
    height_in_meters = ee.Number(pixelHeight).multiply(scale)

    # convert meters to degrees approximately for longitude and latitude
    # this is done in an approximate way, using a conversion factor for degrees
    dx = width_in_meters.divide(111320)
    dy = height_in_meters.divide(110540)

    # create sequences for longitude and latitude to generate grid points
    longs = ee.List.sequence(xmin, xmax, dx)
    lats = ee.List.sequence(ymax, ymin, dy.multiply(-1))  # ensure decrement for latitude

    # helper function to create grid rectangles
    def make_rects_lon(lon):
        lon = ee.Number(lon)  # lon must be ee.Number for arithmetic operations
        def make_rects_lat(lat):
            lat = ee.Number(lat)  # same for lat
            rect = ee.Geometry.Rectangle([lon, lat, lon.add(dx), lat.add(dy)])
            return ee.Feature(rect)

        return lats.map(make_rects_lat)

    # make the grid and flatten the resulting list of lists
    rects = longs.map(make_rects_lon).flatten()
    grid = ee.FeatureCollection(rects)

    return grid