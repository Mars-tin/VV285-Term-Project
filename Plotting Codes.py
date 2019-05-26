#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from matplotlib import cm

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


t = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
p = np.linspace(0, 2, 20)

x = p * np.cos(t)
y = p * np.sin(t) + 2 * p

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')

cmap = cm.bone
c = np.linspace(0.2, 0.8, 20)
[ax.plot(x[:, i], y[:, i], color=cmap(c[i]), linewidth=2) for i in range(20)]
plt.show()


# In[3]:


t = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
p = np.linspace(0, 2, 200)

x = p * np.cos(t)
y = p * np.sin(t) + 2 * p

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')

cmap = cm.bone
c = np.linspace(0.2, 0.8, 200)
[ax.plot(x[:, i], y[:, i], color=cmap(c[i]), linewidth=2) for i in range(200)]

x = np.linspace(-2, 2, 200)
y = np.sqrt(3) * np.abs(x)
ax.plot(x, y, color='r', ls='-')

plt.show()


# In[4]:


sp.init_printing()
x, y, theta = sp.symbols('x y theta')

Eq1 = sp.Eq(x**2 + y**2, 1)
Eq2 = sp.Eq(y - sp.sin(theta), sp.tan(2 * theta) * (x - sp.cos(theta)))
sx = sp.solve(Eq2, x)[0]
sp.simplify(sx.subs(y, 0))


# In[5]:


sp.init_printing()
x, y, theta = sp.symbols('x y theta')

Eq1 = sp.Eq(x**2 + y**2, 1)
Eq2 = sp.Eq(y - sp.sin(theta), sp.tan(3 * theta / 2) * (x - sp.cos(theta)))
sx = sp.solve(Eq2, x)[0]
sp.simplify(sx.subs(y, 0))


# In[6]:


curve = y - sp.sin(theta) - sp.tan(2 * theta) * (x - sp.cos(theta))


# In[7]:


fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.set_aspect('equal')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
t = np.linspace(-1, 1, 800)
theta = np.arcsin(t)
for i in range(theta.size):
    phi = theta[i]
    x = np.linspace(-1, 1, 2)
    y = x * np.tan(2 * phi) + np.sin(phi) - np.cos(phi) * np.tan(2 * phi)
    ax.plot(x, y, color='C0', alpha=0.1, linewidth=2)
ax.plot(np.cos(theta), np.sin(theta), linewidth=2, color='k')

ax = fig.add_subplot(1, 2, 2)
ax.set_aspect('equal')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
theta = np.linspace(-np.pi, np.pi, 800)
for i in range(theta.size):
    phi = theta[i]
    x = np.linspace(-1, 1, 2)
    y = x * np.tan(1.5 * phi) + np.sin(phi) - np.cos(phi) * np.tan(1.5 * phi)
    ax.plot(x, y, color='C0', alpha=0.1, linewidth=2)
ax.plot(np.cos(theta), np.sin(theta), linewidth=2, color='k')
plt.show()


# In[8]:


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
t = np.linspace(-1, 1, 16)
theta = np.arcsin(t)
for i in range(theta.size):
    phi = theta[i]
    x = np.linspace(-1, 1, 2)
    y = x * np.tan(2 * phi) + np.sin(phi) - np.cos(phi) * np.tan(2 * phi)
    ax.plot(x, y, color='C0', linewidth=1)
    ax.plot([-1, np.cos(phi)], [np.sin(phi), np.sin(phi)], color='C0')
ax.plot()

theta = np.linspace(-np.pi/2, np.pi/2, 100)
x = (3 * np.cos(theta) - np.cos(3 * theta)) / 4
y = (3 * np.sin(theta) - np.sin(3 * theta)) / 4
ax.plot(x, y, color='r')
ax.plot(np.cos(theta), np.sin(theta), linewidth=2, color='k')


# In[9]:


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
theta = np.linspace(-np.pi, np.pi, 32)
for i in range(theta.size):
    phi = theta[i]
    x = np.linspace(-1, 1, 2)
    y = x * np.tan(1.5 * phi) + np.sin(phi) - np.cos(phi) * np.tan(1.5 * phi)
    ax.plot(x, y, color='C0', linewidth=1)
    ax.plot([-1, np.cos(phi)], [0, np.sin(phi)], alpha=0.5, color='C0')
ax.plot()

theta = np.linspace(-np.pi, np.pi, 800)
x = -2 / 3 * np.cos(theta) * (1 + np.cos(theta)) + 1 / 3
y = 2 / 3 * np.sin(theta) * (1 + np.cos(theta))
ax.plot(x, y, color='r')
ax.plot(np.cos(theta), np.sin(theta), linewidth=2, color='k')


# In[10]:


def y(x):
    return x * np.tan(2 * phi) + np.sin(phi) - np.cos(phi) * np.tan(2 * phi)


# In[11]:


from matplotlib.patches import Arc

def get_angle(p0, p1=np.array([0,0]), p2=None):
    ''' compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    '''
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

def rotation_transform(theta):
    ''' rotation matrix given theta
    Inputs:
        theta    - theta (in degrees)
    '''
    theta = np.radians(theta)
    A = [[np.math.cos(theta), -np.math.sin(theta)],
         [np.math.sin(theta), np.math.cos(theta)]]
    return np.array(A)


# In[12]:


def add_corner_arc(ax, line, radius=.7, color=None, text=None, text_radius=.5, text_rotatation=0, **kwargs):
    ''' display an arc for p0p1p2 angle
    Inputs:
        ax     - axis to add arc to
        line   - MATPLOTLIB line consisting of 3 points of the corner
        radius - radius to add arc
        color  - color of the arc
        text   - text to show on corner
        text_radius     - radius to add text
        text_rotatation - extra rotation for text
        kwargs - other arguments to pass to Arc
    '''

    lxy = line.get_xydata()

    if len(lxy) < 3:
        raise ValueError('at least 3 points in line must be available')

    p0 = lxy[0]
    p1 = lxy[1]
    p2 = lxy[2]

    width = np.ptp([p0[0], p1[0], p2[0]])
    height = np.ptp([p0[1], p1[1], p2[1]])

    n = np.array([width, height]) * 1.0
    p0_ = (p0 - p1) / n
    p1_ = (p1 - p1)
    p2_ = (p2 - p1) / n 

    theta0 = -get_angle(p0_, p1_)
    theta1 = -get_angle(p2_, p1_)

    if color is None:
        # Uses the color line if color parameter is not passed.
        color = line.get_color() 
    arc = ax.add_patch(Arc(p1, width * radius, height * radius, 0, theta0, theta1, color=color, **kwargs))

    if text:
        v = p2_ / np.linalg.norm(p2_)
        if theta0 < 0:
            theta0 = theta0 + 360
        if theta1 < 0:
            theta1 = theta1 + 360
        theta = (theta0 - theta1) / 2 + text_rotatation
        pt = np.dot(rotation_transform(theta), v[:,None]).T * n * text_radius
        pt = pt + p1
        pt = pt.squeeze()
        ax.text(pt[0], pt[1], text,         
                horizontalalignment='left',
                verticalalignment='top',)

    return arc


# In[13]:


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_xlim(-0.8, 1.2)
ax.set_ylim(-1, 1)
ax.axis('off')

ax.scatter(0, 0, marker='.', color='k')
ax.text(0.05, 0, r'$O$')

theta = np.linspace(-np.pi/2, np.pi/2, 200)
phi = np.pi/6
ax.scatter(np.cos(phi), np.sin(phi), marker='.', color='k')
ax.text(np.cos(phi) + 0.05, np.sin(phi) + 0.05, r'$R$')

x = np.linspace(-1, np.cos(phi), 2)

ax.arrow(0, np.sin(phi), 0.1, 0, shape='full', lw=0, length_includes_head=True, head_width=.05)
ax.arrow(.4, y(.4), -0.1, y(.4) - y(.5), shape='full', lw=0, length_includes_head=True, head_width=.05)

ax.plot(x, [np.sin(phi), np.sin(phi)])
ax.plot(x, y(x), color='C0')

ax.plot([0, np.cos(phi)], [0, np.sin(phi)], color='k', ls='--')

ax.plot(np.cos(theta), np.sin(theta), color='k')
plt.show()


# In[14]:


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_xlim(-0.8, 1.2)
ax.set_ylim(-1, 1)
ax.axis('on')
ax.grid('off')

ax.scatter(0, 0, marker='.', color='k')
ax.text(0.05, -0.1, r'$O$')

Bx = -np.sin(phi) / np.tan(2 * phi) + np.cos(phi)
ax.scatter(Bx, 0, color='k', marker='.')
ax.text(Bx + 0.05, 0, r'$B$')

theta = np.linspace(-np.pi/2, np.pi/2, 200)
phi = np.pi/6
ax.scatter(np.cos(phi), np.sin(phi), marker='.', color='k')
ax.text(np.cos(phi) + 0.05, np.sin(phi) + 0.05, r'$R$')

x = np.linspace(-1, np.cos(phi), 2)

ax.arrow(0, np.sin(phi), 0.1, 0, shape='full', lw=0, length_includes_head=True, head_width=.05)
ax.arrow(.4, y(.4), -0.1, y(.4) - y(.5), shape='full', lw=0, length_includes_head=True, head_width=.05)

line3, = ax.plot([np.cos(phi), Bx, 0], [np.sin(phi), 0, 0], color='k', alpha=0.5)
line1, = ax.plot([-1, np.cos(phi), -1], [np.sin(phi), np.sin(phi), y(-1)])
line2, = ax.plot([np.cos(phi), 0, -0.01], [np.sin(phi), 0, 0], color='k', ls='--')
ax.plot([-1, 0], [0, 0], color='k', alpha=0.5)

add_corner_arc(ax, line1, 0.2, color='k')
ax.text(np.cos(phi)-0.3, np.sin(phi)-0.1, r'$\alpha$')
ax.text(np.cos(phi)-0.25, np.sin(phi)-0.25, r'$\alpha$')

add_corner_arc(ax, line3, 0.2, color='k')
ax.text(Bx - 0.1, 0.1, r'$\beta$')

add_corner_arc(ax, line2, 0.2, color='k')
ax.text(-0.1, 0.1, r'$\theta$')

ax.plot(np.cos(theta), np.sin(theta), color='k')
plt.show()


# In[15]:


def y(x):
    return x * np.tan(1.5 * phi) + np.sin(phi) - np.cos(phi) * np.tan(1.5 * phi)


# In[16]:


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.axis('off')

ax.scatter(0, 0, marker='.', color='k')
ax.text(0.05, 0, r'$O$')

theta = np.linspace(-np.pi, np.pi, 200)
phi = np.pi/6
(Rx, Ry) = (np.cos(phi), np.sin(phi))

ax.scatter(np.cos(phi), np.sin(phi), marker='.', color='k')
ax.text(np.cos(phi) + 0.05, np.sin(phi) + 0.05, r'$R$')
ax.text(-1, 0, r'$S$')
ax.text(-0.5, y(-0.5), r'B')

Ax = -np.sin(phi) / np.tan(3 * phi / 2) + np.cos(phi)

ax.arrow(-1, 0, (Rx + 1) / 2, Ry / 2, shape='full', lw=0, length_includes_head=True, head_width=.05)
ax.arrow(Rx, Ry, -(Rx - Ax), -(Ry - y(Ax)), shape='full', lw=0, length_includes_head=True, head_width=.05)


line1, = ax.plot([-1, Rx, -0.5], [0, Ry, y(-0.5)])
line2, = ax.plot([0, Rx], [0, Ry], color='k', ls='--')

ax.plot(np.cos(theta), np.sin(theta), color='k')
plt.show()


# In[17]:


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.axis('on')
ax.grid('off')

ax.scatter(0, 0, marker='.', color='k')
ax.text(0.05, -0.1, r'$O$')

theta = np.linspace(-np.pi, np.pi, 200)
phi = np.pi/6
ax.scatter(np.cos(phi), np.sin(phi), marker='.', color='k')
ax.text(np.cos(phi) + 0.05, np.sin(phi) + 0.05, r'$R$')
ax.text(-0.9, 0.1, r'$S$')
ax.text(-0.5, y(-0.5), r'$B$')
ax.text(Ax + 0.1, 0, r'$A$')

ax.arrow(-1, 0, (Rx + 1) / 2, Ry / 2, shape='full', lw=0, length_includes_head=True, head_width=.05)
ax.arrow(Rx, Ry, -(Rx - Ax), -(Ry - y(Ax)), shape='full', lw=0, length_includes_head=True, head_width=.05)

(Rx, Ry) = (np.cos(phi), np.sin(phi))

line4, = ax.plot([0, -1, -0], [0, 0, Ry / (Rx + 1)], color='k', alpha=0.5)
line3, = ax.plot([Rx, Ax, 0], [Ry, 0, 0], color='k', alpha=0.5)
line1, = ax.plot([-1, Rx, -0.5], [0, Ry, y(-0.5)])
line2, = ax.plot([Rx, 0, -.1], [Ry, 0, 0], color='k', ls='--')

add_corner_arc(ax, line1, 0.3, color='k')
ax.text(Rx-0.4, Ry-0.15, r'$\alpha$')
ax.text(Rx-0.35, Ry-0.25, r'$\alpha$')

add_corner_arc(ax, line4, 0.2, color='k')
ax.text(-0.8, 0, r'$\alpha$')

add_corner_arc(ax, line3, 0.2, color='k')
ax.text(Ax-0.1, 0.1, r'$\beta$')

add_corner_arc(ax, line2, 0.2, color='k')
ax.text(-0.1, 0.1, r'$\theta$')

ax.plot(np.cos(theta), np.sin(theta), color='k')
plt.show()


# In[ ]:




