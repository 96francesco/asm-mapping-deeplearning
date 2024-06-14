import ee

ee.Authenticate()
ee.Initialize(project='asm-drc')

def delete_all_assets(project_id):
    """
    Delete all assets in the GEE project
    """
    asset_list = ee.data.listAssets({'parent': f'projects/{project_id}/assets/'})['assets']
    for asset in asset_list:
        try:
            ee.data.deleteAsset(asset['name'])
            print(f"Deleted asset: {asset['name']}")
        except Exception as e:
            print(f"Failed to delete asset {asset['name']}: {e}")

delete_all_assets('asm-drc')
