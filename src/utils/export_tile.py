import ee

def export_tile(image, tile, asset_id):
    """
    Export a tile as GEE asset
    """
    description = asset_id.replace("/", "_")
    region = tile.geometry().bounds()
    task = ee.batch.Export.image.toAsset(
        image=image.clip(region),
        description=description,
        assetId=f'projects/asm-drc/assets/{asset_id}',
        scale=4.77,
        region=region.getInfo()['coordinates'],
        maxPixels=1e13
    )
    task.start()
    return task