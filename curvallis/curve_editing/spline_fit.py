import numpy as np
import math
import matplotlib as mpl
import curve_fitters
import io
import lines
import smoothers
import regions
from matplotlib import pyplot as plt
from scipy import interpolate

#Define parser arguments
def define_args(parser):
    #cmto add argument for spline fit
    parser.add_argument(
        '--do_spline',
        action='store_true',
        help='Calculate cubic spline fit '
             '[default: %(default)s]')    
    parser.add_argument(
        '--num_fit_points', action='store', type=int,
        help='Calculate this many points in the spline fit curve when writing the '
            'curve to a file [default: %(default)s]', metavar='<count>')
    parser.add_argument(
        '--spline_bound',
        action='append',
        nargs='+', type=float,
        help='X-values defining boundaries for spline fit are supposed to go in sequential order'
            '[default: Evenly Spaced]',
        metavar='<bound>')
    parser.add_argument(
        "--xlabel", type=str, 
        help="X Label for plot of data and spline fit.")
    parser.add_argument(
         "--ylabel", type=str, 
         help="Y Label for plot of data and spline fit.")
    parser.add_argument(
         '--print_E2P', action='store_true', 
         help='Print derivative P = rho^2*dE/d(rho)')
    parser.add_argument(
         '--print_P2B', action='store_true', 
         help='Print derivative B = rho*dP/d(rho) ')    
    parser.set_defaults(
        do_spline=False,    
        num_fit_points=5,
        spline_bound = list(),
        xlabel='X',
        ylabel='Y',
        print_E2P=False,
        print_P2B=False,
    )

class Spline_Fit(object):
    """ Manages all the cubic spline curve fitting.
    """
    def __init__(self, ax, args, input_data_sets, xy_limits, io_manager):
        self._args = args
        self._ax = ax
        self._io_manager = io_manager
        self._data_sets = input_data_sets.get_copy()
        self._x_max = xy_limits.x_max
        self._x_min = xy_limits.x_min
        self._first_time = True
        self._fitpoints = list()
        self._derivpoints = list()
        self._spline = None
        self._xp = list()

        self.fit_curve = lines.Line(ax, lines.line_attributes['fit_curve'])
        self.derivative_curve = lines.Line(ax, lines.line_attributes['derivative'])

    #Function to draw spline curve fit and derivative curve
    def draw(self):
        self.fit_curve.draw()
        self.derivative_curve.draw()

    #Function to print the equation of the spline curve fit
    def print_equation(self,xp1,spline1):
        for i in range(xp1.shape[0] - 1):
            sx = np.linspace(xp1[i], xp1[i+1], self._args.num_fit_points)
            sy = spline1(sx)
            print ("==================================================================================================================================")
            print ('REGION %s from %.2E to %.2E: (%.2E)*(x-%.2E)^3 + (%.2E)*(x-%.2E)^2 + (%.2E)*(x-%.2E) + %.2E' %
                (i, xp1[i], xp1[i+1], spline1.c[0,i], xp1[i], spline1.c[1,i], xp1[i], spline1.c[2,i], xp1[i], spline1.c[3,i]))

            print ('REGION %s Derivative from %.2E to %.2E: 3*(%.2E)*(x-%.2E)^2 + 2*(%.2E)*(x-%.2E) + (%.2E)' %
                (i, xp1[i], xp1[i+1], spline1.c[0,i], xp1[i], spline1.c[1,i], xp1[i], spline1.c[2,i]))

    #Function to filter points for duplicate entries
    def filter_points(self,inpts):
        # Remove any duplicate fit points at region boundaries
        filtered_points = [inpts[0]]
        for i in range(1, len(inpts)):
            if (not np.allclose(inpts[i],inpts[i-1])):
                filtered_points.append(inpts[i])
        return filtered_points            

    #Function to check if there exist two points with same x-value
    def check_double_value(self,inpts):
        # Remove any duplicate fit points at region boundaries
        for i in range(1, len(inpts)):
            if np.isclose(inpts[i][0],inpts[i-1][0]):
                print ('Warning: Double Valued X: %s, %s' % (inpts[i][0],inpts[i-1][0]))
                return True
        return False

    #Function to write output file with points
    def write_output(self,outfile,inpts):
        with open(outfile,'w') as out_file:
            print ("Writing points to %s" % outfile)
            for i in range(0,len(inpts)):
                wx,wy = zip(*inpts)
                wx1 = list(wx)
                wy1 = list(wy)
                out_file.write('% .15E % .15E\n' % (wx1[i],wy1[i]))
            print ("DONE Writing points to %s" % outfile)

    #Function to write derivative points and terms and print equation for spline fit and derivative
    def write_derivative_files(self):
        dxt, dyt = zip(*self._derivpoints)
        dx = list(dxt)
        dy = list(dyt)
        
        self.write_output("derivative_fit.dat",self._derivpoints)

        #Print out E2P derivative term
        if self._args.print_E2P:
            temp = list()
            for i in range(0,len(self._derivpoints)):
                temp.append(dy[i]*dx[i]**2)
            e2ppoints = zip(dx,temp)
            e2pdata = sorted(e2ppoints,key=lambda x: float(x[0]))
            self.write_output('E2P.dat',e2pdata)
        
        #Print out P2B derivative term
        if self._args.print_P2B:
            temp = list()
            for i in range(0,len(self._derivpoints)):
                temp.append(dy[i]*dx[i])
            p2bpoints = zip(dx,temp)
            p2bdata = sorted(p2bpoints,key=lambda x: float(x[0]))
            self.write_output('P2B.dat',p2bdata)

        #Print Equation
        print ("Printing Equations for Fit and Derivative Curves:")
        self.print_equation(self._xp,self._spline)

    #Function to return spline fit points
    def get_fit_curve_points(self):
        return self._fitpoints

    #Function to return derivative points
    def get_derivative_points(self):
        return self._derivpoints

    #Main function to plot the spline. Takes in data_sets as argument.
    def plot_spline_curve(self, data_sets):
        all_data = sorted(data_sets.get_only_set(),key=lambda x: float(x[0]))
        x0, y0 = zip(*all_data)
        x_data = list(x0)
        y_data = list(y0)        
        
        #Check if regions are reasonable
        if len(self._args.spline_bound) != 0:
            if max(self._args.spline_bound[0]) > self._x_max:
                raise RuntimeError("Max region bound > max x_value")
            if min(self._args.spline_bound[0]) < self._x_min:
                raise RuntimeError("Min region bound < min x_value")

        #Check double valued X in sorted data. Exit with Error if so.
        if self.check_double_value(all_data):
            raise RuntimeError("Double valued X to input file: Not Allowed")

        #Check if spline boundaries are specified in config file; if not, use all x data points as regions
        if len(self._args.spline_bound) < 1:
            xp = np.asarray(x_data)
            spline = interpolate.CubicSpline(x_data,y_data)
        else:
            model = interpolate.CubicSpline(x_data,y_data)
            xp = np.asarray(self._args.spline_bound[0])
            yp = model(xp)
            spline = interpolate.CubicSpline(xp,yp)

        all_x = list()
        all_y = list()
        all_dy = list()

        #Fit spline model to each spline boundary, print out equation
        for i in range(xp.shape[0] - 1):
            segment_x = np.linspace(xp[i], xp[i+1], self._args.num_fit_points)
            segment_y = spline(segment_x)
            all_x.extend(segment_x.tolist())
            all_y.extend(segment_y.tolist())

            #Calculate derivative and print out analytical derivative formula
            for xi in segment_x:
                derivative = 3*spline.c[0,i]*(xi-xp[i])**2 + 2*spline.c[1,i]*(xi-xp[i]) + spline.c[2,i]
                all_dy.append(derivative)            

        #Only print out equation if this is the first time plot_spline_curve is called.
        if self._first_time:
            self.print_equation(xp,spline)
            self._first_time = False

        fit_points = zip(all_x,all_y)
        derivative_points = zip(all_x,all_dy)

        #Filter fit points and derivative to remove duplicate entries
        filtered_points = self.filter_points(fit_points)
        filtered_derivative = self.filter_points(derivative_points)        

        #Store spline fit, fit points, derivative points and x-values for future use
        self._fitpoints = filtered_points
        self._derivpoints = filtered_derivative
        self._spline = spline
        self._xp = xp

#        fx0,fy0 = zip(*filtered_points)
#        fx = list(fx0)
#        fy = list(fy0)

#        dx0,dy0 = zip(*filtered_derivative)
#        dx = list(dx0)
#        dy = list(dy0)

        #set up a plot GUI
        self._ax.set_xlabel(self._args.xlabel,fontsize=16)
        self._ax.set_ylabel(self._args.ylabel,fontsize=16)
        self._ax.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout(True)

#        min_x = min(all_x)
#        max_x = max(all_x)

        #Plot fit curve and derivative curve
        if self.fit_curve._id == None:
            self.fit_curve.plot_xy_data(filtered_points,animated=True)
        else:
            self.fit_curve.set_xy_data(filtered_points)

        if self.derivative_curve._id == None:
            self.derivative_curve.plot_xy_data(filtered_derivative,animated=True)
        else:
            self.derivative_curve.set_xy_data(filtered_derivative)
