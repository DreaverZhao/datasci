import numpy as np
import scipy.stats
import pandas
import matplotlib.pyplot as plt
import matplotlib.patches
import imageio.v2 as imageio
import numbers
from IPython.display import clear_output

map_image = imageio.imread('https://www.cl.cam.ac.uk/teaching/2223/DataSci/data/voronoi-map-goal-16000-shaded.png')
localization = pandas.read_csv('https://www.cl.cam.ac.uk/teaching/2223/DataSci/data/localization_2022.csv')
localization.sort_values(['id','t'], inplace=True)

# Pull out observations for the animal we want to track
observations = localization.loc[localization.id==0, ['r','g','b']].values

df = localization

# fig,(ax,ax2) = plt.subplots(2,1, figsize=(4,5), gridspec_kw={'height_ratios':[4,.5]})
# ax.imshow(map_image.transpose(1,0,2), alpha=.5)
# w,h = map_image.shape[:2]
# ax.set_xlim([0,w])
# ax.set_ylim([0,h])

# for i in range(1,5):
#     ax.plot(df.loc[df.id==i,'x'].values, df.loc[df.id==i,'y'].values, lw=1, label=i)
# ax.axis('off')
# ax.legend()
# ax.set_title('Animals 1--4, GPS tracks')

# ax2.bar(np.arange(len(observations)), np.ones(len(observations)), color=observations, width=2)
# ax2.set_xlim([0,len(observations)])
# ax2.set_yticks([])
# ax2.set_title('Animal id=0, camera only')

# plt.tight_layout()
# plt.show()

W,H = map_image.shape[:2]
M = num_particles = 2000

# Empirical representation of the distribution of X0
δ0 = np.column_stack([np.random.uniform(0,W-1,size=M), np.random.uniform(0,H-1,size=M), np.ones(M)/M])
def show_particles(particles, ax=None, s=1, c='red', alpha=.5):
    # Plot an array of particles, with size proportional to weight.
    # (Scale up the sizes by setting s larger.)
    if ax is None:
        fig,ax = plt.subplots(figsize=(2.5,2.5))
    ax.imshow(map_image.transpose(1,0,2), alpha=alpha, origin='lower')
    w,h = map_image.shape[:2]
    ax.set_xlim([0,w])
    ax.set_ylim([0,h])
    w = particles[:,2]
    ax.scatter(particles[:,0],particles[:,1], s=w/np.sum(w)*s, color=c)
    ax.axis('off')
# fig,ax = plt.subplots(figsize=(4,4))
# show_particles(δ0, s=400, ax=ax)
# ax.set_title('$X_0$')
# plt.show()

# y0 = observations[0]
# print(f"First observation: rgb = {y0}")

def patch(im, xy, size=3):
    s = (size-1) / 2
    nx,ny = np.meshgrid(np.arange(-s,s+1), np.arange(-s,s+1))
    nx,ny = np.stack([nx,ny], axis=0).reshape((2,-1))
    neighbourhood = np.row_stack([nx,ny])
    w,h = im.shape[:2]
    neighbours = neighbourhood + np.array(xy).reshape(-1,1)
    neighbours = nx,ny = np.round(neighbours).astype(int)
    nx,ny = neighbours[:, (nx>=0) & (nx<w) & (ny>=0) & (ny<h)]
    patch = im[nx,ny,:3]
    return np.mean(patch, axis=0)/255
# loc = δ0[0,:2]
# print(f"First particle is at {loc}")

# col = patch(map_image, loc, size=3)
# print(f"Map terrain around this particle: rgb = {col}")

# get data to fit the model
locations, color = df.loc[df.id>0,['x','y']].values, df.loc[df.id>0,['r','g','b']].values
y0 = np.array([patch(map_image, loc) for loc in locations])
y0, y = matplotlib.colors.rgb_to_hsv(y0), matplotlib.colors.rgb_to_hsv(color)

def logPr(x,p):
    (μ,τ) = p
    return np.log(scipy.stats.norm.pdf(x,loc=μ,scale=np.exp(τ)))

train_data = np.split(y - y0,3,axis=1)
params = [scipy.optimize.fmin(lambda p: -np.sum(logPr(train_data[i],p)), [0,0.1]) for i in range(3)]

# fig,axs = plt.subplots(3,1, figsize=(10,4), sharex=True)
# for i in range(3):
#     x = np.linspace(-0.5,1.0,200)
#     y = np.exp(logPr(x, params[i]))
#     axs[i].plot(x, y, color='black')
#     axs[i].hist(train_data[i], bins = np.linspace(-0.5,1.0,80), density=True, ec='steelblue', fc='steelblue', alpha=.5)
# plt.plot()

# probability of observing y at location loc
def pr(y, loc):
    y0,y = matplotlib.colors.rgb_to_hsv([patch(map_image,loc),y])
    e = y - y0
    return np.exp(np.sum([logPr(e[i],params[i]) for i in range(0,3)]))

y0 = observations[0]
w = np.array([pr(y0, (x,y)) for x,y,_ in δ0])
π0 = np.copy(δ0)
π0[:,2] = w / np.sum(w)

fig,(axδ,axπ) = plt.subplots(1,2, figsize=(8,4), sharex=True, sharey=True)
show_particles(δ0, ax=axδ, s=600)
show_particles(π0, ax=axπ, s=600)
axπ.add_patch(matplotlib.patches.Rectangle((0,0),100,100,color=y0))
axπ.text(50,50,'$y_0$', c='white', ha='center', va='center', fontsize=14)
axδ.set_title('$X_0$')
axπ.set_title('$(X_0|y_0)$')
plt.show()

# calculate the average location change between consecutive observations in lenth
diff = []
for i in range(1,len(locations)):
    diff.append(np.linalg.norm(locations[i]-locations[i-1]))
diff = np.mean(diff)
print(f"Average location change between consecutive observations: {diff:.2f}")

def walk(loc):
    # randomly choose a direction in [0, 2π)
    dir = np.random.uniform(0,2*np.pi)
    # randomly choose a distance with mean diff
    dist = np.random.exponential(np.linalg.norm(diff))
    # calculate new location
    new_loc = loc + dist*np.array([np.cos(dir), np.sin(dir)])
    # if new location is outside the map, limit it to the map boundary
    if new_loc[0]<0 or new_loc[0]>W-1 or new_loc[1]<0 or new_loc[1]>H-1:
        new_loc = np.clip(new_loc, 0, [W-1,H-1])
    return new_loc

# Sanity check
loc = π0[0,:2]
loc2 = walk(loc)
assert len(loc2)==2 and isinstance(loc2[0], numbers.Number) and isinstance(loc2[1], numbers.Number)
assert loc2[0]>=0 and loc2[0]<=W-1 and loc2[1]>=0 and loc2[1]<=H-1

δ1 = np.copy(π0)
for i in range(len(δ1)):
    δ1[i,:2] = walk(δ1[i,:2])
fig,ax = plt.subplots(figsize=(4,4))
show_particles(π0, ax=ax, s=4000, c='blue', alpha=.25)
show_particles(δ1, ax=ax, s=4000, c='red', alpha=.25)
ax.set_xlim([200,400])
ax.set_ylim([100,300])
plt.show()

particles = np.copy(π0)

for n,obs in enumerate(observations[:50]):
    # Compute δ, the locations after a movement step
    for i in range(num_particles):
        particles[i,:2] = walk(particles[i,:2])
    # Compute π, the posterior after observing y
    w = particles[:,2]
    w *= np.array([pr(obs, (px,py)) for px,py,_ in particles])
    particles[:,2] = w / np.sum(w)

    # Plot the current particles
    fig,ax = plt.subplots(figsize=(3.5,3.5))
    show_particles(particles, ax, s=20)
    ax.set_title(f"Timestep {n+1}")
    plt.show()
    clear_output(wait=True)
