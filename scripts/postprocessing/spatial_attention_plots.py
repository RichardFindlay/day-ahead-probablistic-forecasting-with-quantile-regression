import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas
import contextily as ctx
from pickle import load
from matplotlib.animation import FuncAnimation


# plot spatial attention
def plot_spatial_predictions(spatial_data, title, height_scale, width_scale, frame_num):

	fig = plt.figure(figsize=[8,10])  # a new figure window
	ax_set = fig.add_subplot(1, 1, 1)

	# create baseline map
	# spatial data on UK basemap
	df = pd.DataFrame({
		'LAT': [49.78, 61.03],
		'LON': [-11.95, 1.55],
	})

	geo_df = geopandas.GeoDataFrame(df, crs = {'init': 'epsg:4326'}, 
			geometry=geopandas.points_from_xy(df.LON, df.LAT)).to_crs(epsg=3857)

	ax = geo_df.plot(
		figsize= (8,10),
		alpha = 0,
		ax=ax_set,
	)

	plt.title(title)
	ax.set_axis_off()

	# add basemap
	url = 'http://tile.stamen.com/terrain/{z}/{x}/{y}.png'
	zoom = 10
	xmin, xmax, ymin, ymax = ax.axis()
	basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, source=url)
	ax.imshow(basemap, extent=extent, interpolation='gaussian')
	attn_over = np.resize(spatial_data[0], (height_scale, width_scale))
	
	gb_shape = geopandas.read_file("../../data/raw/_mapping/shapefiles/GBR_adm/GBR_adm0.shp").to_crs(epsg=3857)
	irl_shape = geopandas.read_file("../../data/raw/_mapping/shapefiles/IRL_adm/IRL_adm0.shp").to_crs(epsg=3857)
	gb_shape.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5, alpha=0.4)
	irl_shape.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5, alpha=0.4)
	overlay = ax.imshow(attn_over, cmap='viridis', alpha=0.5, extent=extent)
	# ax.axis((xmin, xmax, ymin, ymax))
	txt  = fig.text(.5, 0.09, '', ha='center')

	
	def update(i):
		spatial_over = np.resize(spatial_data[i], (height_scale, width_scale))
		print(spatial_over.shape)
		# overlay = ax.imshow(spatial_over, cmap='viridis', alpha=0.5, extent=extent)
		overlay.set_data(spatial_over)
		txt.set_text(f"Timestep: {i}")
		# plt.cla()

		return [overlay, txt]


	animation_ = FuncAnimation(fig, update, frames=frame_num, blit=False, repeat=False)
	# plt.show(block=True)	
	animation_.save(f'{title}_animation.gif', writer='imagemagick')



# define model type to plot
model_type = 'solar'

idx = 0

# load spatial attention data
# save results - forecasted spatial attention matrix
with open(f'../../results/{model_type}/seq2seq+temporal+spatial/spatial_attention_data_{model_type}.pkl', 'rb') as spatial_file:
	spatial_data = load(spatial_file)


# grab relevant example
spatial_data = spatial_data['0.5'][idx,:,:]

spatial_data = np.transpose(spatial_data)

print(spatial_data.shape)
print(spatial_data[30, :])

# exit()


# call plot function
plot_spatial_predictions(spatial_data=spatial_data, title='Solar Spatial Attention', height_scale=16, width_scale=20, frame_num=48)