import ee

def batch_delete_assets(asset_ids):
    for asset_id in asset_ids:
        try:
            ee.data.deleteAsset("projects/asm-drc/assets/" + asset_id)
            print(f"Deleted asset {asset_id}")
        except ee.ee_exception.EEException as e:
            if 'Permission denied' in str(e):
                print(f"Permission denied for deleting asset {asset_id}: {e}")
            elif 'does not exist' in str(e):
                print(f"Asset does not exist: {asset_id}")
            else:
                print(f"Failed to delete asset {asset_id}: {e}")