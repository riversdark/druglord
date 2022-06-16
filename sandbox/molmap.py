import molmap
# Define your molmap
mp_name = './descriptor.mp'
mp = molmap.MolMap(ftype='fingerprint', fmap_type='scatter',
                   split_channels = True, metric='cosine', var_thr=1e-4)
# Fit your molmap
mp.fit(method = 'umap', verbose = 2)
mp.transform('c1ccc2c(c1)ccc3c2ccc4c5ccccc5ccc43')
mp.save(mp_name) 