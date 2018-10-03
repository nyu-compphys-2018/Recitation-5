import numpy as np
import matplotlib.pyplot as plt

class Burger:

    a = 0.0
    b = 1.0
    Nx = 10
    dx = None
    t = 0.0
    cfl = 0.5

    x = None
    u = None

    def __init__(self, Nx, a, b, t, cfl):

        self.Nx = Nx
        self.a = a
        self.b = b
        self.t = t
        self.cfl = cfl

        self.dx = (b-a)/float(Nx)
        self.u = np.zeros(Nx)
        self.x = a + self.dx*( np.arange(Nx) + 0.5 )

    def setInitConditions(self, f, *args):
        self.u = f( self.x, *args )

    def evolve(self, tfinal):
        
        #Get dt, and make sure don't overrun tfinal
        while self.t < tfinal:
            dt = self.get_dt()
            if self.t+dt > tfinal:
                dt = tfinal - self.t

            #Calculate Fluxes
            udot = self.LU()

            #update u
            self.u += dt*udot
            self.t += dt


    def get_dt(self):
        return self.cfl * self.dx / np.max( np.fabs(self.u) )

    def LU(self):

        ap = np.empty(self.Nx-1)
        am = np.empty(self.Nx-1)
        for i in range(self.Nx-1):
            ap[i] = max(0, self.u[i], self.u[i+1] )
            am[i] = max(0, -self.u[i], -self.u[i+1] )

        F = 0.5*self.u*self.u

        FL = F[:-1]
        FR = F[1:]
        UL = self.u[:-1]
        UR = self.u[1:]

        FHLL = (ap*FL + am*FR - ap*am*(UR-UL))/(ap+am)
        LU = np.zeros(self.Nx)
        LU[1:-1] = -(FHLL[1:] - FHLL[:-1])/self.dx

        return LU

    def plot(self, ax=None, filename=None ):

        if ax is None:
            fig,ax = plt.subplots(1,1)
        else:
            fig = ax.get_figure()

        ax.plot( self.x, self.u, 'k+' )
        ax.set_xlabel(r'$X$' )
        ax.set_ylabel(r'$Y$' )
        ax.set_title("t = {0:f}".format( self.t ) )

        
        return (ax)

def gauss(x, x0, sigma ):
    return np.exp( -(x-x0)*(x-x0)/2/sigma/sigma )

b = Burger( 1500, -3, 4, 0, 0.5 )
b.setInitConditions( gauss, 0.0, 1.0 )
ax = b.plot()
b.evolve(1.0)
b.plot(ax)
b.evolve(2.0)
b.plot(ax)
b.evolve(3.0)
b.plot(ax)

plt.show()
















