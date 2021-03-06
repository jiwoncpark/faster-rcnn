import numpy as np
from skimage.draw import line_aa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
#%matplotlib inline

def make_shower(args):

    img = np.zeros(shape=(args['nx'], args['ny']), dtype=int)

    # randomly generate starting point
    # note there is a lmax buffer on the boundaries of canvas
    # so that shower doesn't fall off the image
    vx, vy = np.random.randint(args['lmax'],args['nx']-args['lmax']), np.random.randint(args['lmax'],args['ny']-args['lmax'])
    theta0 = np.random.uniform(2.*np.pi) # central angle of shower

    # randomly generate nlines endpoints such that the lines fall
    # within around dtheta of theta0
    if args['dtheta'] < 0:
        dtheta = np.random.uniform(0.5*np.pi)
        thetas = np.random.normal(loc=theta0, scale=dtheta, size=(args['nlines'], 1))
    else:
        dtheta = args['dtheta']
        thetas = np.linspace(theta0-dtheta, theta0+dtheta, args['nlines']).reshape(args['nlines'], 1)
        #thetas = np.random.uniform(low=0., high=2.*dtheta, size=(args['nlines'], 1))
        
    lengths = np.random.uniform(low=args['lmin'], high=args['lmax'], size=(args['nlines'], 1))

    # draw shower lines
    for pos in np.hstack(((vx+lengths*np.cos(thetas)+0.5).astype(int), (vy+lengths*np.sin(thetas)+0.5).astype(int))):
        rr, cc, _ = line_aa(vx, vy, pos[0], pos[1])
        img[rr, cc] = 1

    # randomly set pixels to 0
    indices0 = np.random.choice([0, 1], p=[args['keep_prob'], 1-args['keep_prob']], size=img.shape).astype(np.bool)
    #indices0 = np.random.randint(0,2,size=img.shape).astype(np.bool)
    indices1 = np.ones(img.shape)
    indices1[vx-args['keep']:vx+args['keep'], vy-args['keep']:vy+args['keep']] = 0
    img[np.logical_and(indices0, indices1)] = 0

    return img, (vx,vy), (2.*dtheta)

def make_showerset(args):
    blob = {}
    batch_data = np.zeros((args['nimages'], args['nx'], args['ny']))
    batch_angles = np.zeros((args['nimages'],))
    for i in range(args['nimages']):
        img, _, a = make_shower(args)
        batch_data[i,:,:] = img
        batch_angles[i] = a
        if args['out_png']:
            plt.imshow(img)
            plt.savefig('shower_%d.png'%i)
            plt.close()
        if i != 0 and i%20==0: print(i, ' done')
    #batch_data = batch_data.reshape(args['nimages'], args['nx'], args['ny'], 1)
    #np.savetxt('batch_data_shower.txt', batch_data.reshape(-1))
    #np.savetxt('batch_angles.txt', batch_angles)
    blob['data'] = batch_data
    blob['labels'] = np.zeros(args['nimages'],)
    blob['angles'] = batch_angles
    return blob

if __name__ == '__main__':
    '''
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    parser.add_argument('--nx', type=int, default = args_def['nx'],
                        help='x dimension of canvas, 128')
    '''
    args_def = dict(
        nx = 256,        
        ny = 256,      
        nlines = 10,     
        dtheta = np.radians(0.001),
        lmin = 30,
        lmax = 100,
        keep = 7,
        keep_prob = 0.6,
        nimages = 100,
        out_png = False,
    )
    batch_shape = make_showerset(args = args_def)
    print batch_shape
    #img, (x,y), a = make_shower(args_def)
    #print img.shape
    #print x,y
    #print a
