import pandas as pd
import ee
ee.Initialize(project="skytrace-ai")
gaz_collections = {
    'NO2': 'COPERNICUS/S5P/OFFL/L3_NO2',
    'CO': 'COPERNICUS/S5P/OFFL/L3_CO',
    'CH4': 'COPERNICUS/S5P/OFFL/L3_CH4',
    'SO2': 'COPERNICUS/S5P/OFFL/L3_SO2'
}

zones = {
    'Jorf Lasfar (OCP)': [-8.638, 33.105],
    'El Jadida': [-8.500, 33.231],
    'Safi Industrial': [-9.237, 32.299],
    'Mohammedia Port': [-7.383, 33.686],
    'Casablanca Industrial Zone': [-7.603, 33.589],
    'Nouaceur / Midparc': [-7.589, 33.367],
    'Berrechid Industrial Zone': [-7.587, 33.265],
    'Settat Industrial Zone': [-7.620, 33.000],
    'Kenitra Industrial Zone': [-6.578, 34.261],
    'Tangier Med': [-5.503, 35.893],
    'Tangier Automotive City': [-5.912, 35.726],
    'Tetouan Industrial Zone': [-5.362, 35.578],
    'Nador West Med': [-2.928, 35.169],
    'Oujda Industrial Zone': [-1.907, 34.682],
    'Fes Industrial Zone': [-5.003, 34.033],
    'Meknes Industrial Zone': [-5.547, 33.894],
    'Khouribga': [-6.906, 32.881],
    'Beni Mellal Industrial Zone': [-6.349, 32.337],
    'Marrakech Industrial Zone': [-8.008, 31.630],
    'Agadir Industrial Zone': [-9.598, 30.427],
    'Laayoune Industrial Zone': [-13.203, 27.153],
    'Dakhla Industrial Zone': [-15.957, 23.684]
}
results = []

for gaz_name, collection_path in gaz_collections.items():
    data = ee.ImageCollection(collection_path) \
             .filterDate('2026-03-01', '2026-03-31') \
             .mean()
    
    band_name = data.bandNames().get(0) 
    
    for zone_name, coords in zones.items():
        point = ee.Geometry.Point(coords)
        
        val = data.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=1000
        ).get(band_name).getInfo()
        
        results.append({
            'Zone': zone_name,
            'Gaz': gaz_name,
            'Concentration': val
        })

df_final = pd.DataFrame(results)
df_pivot = df_final.pivot(index='Zone', columns='Gaz', values='Concentration')

print(df_pivot)