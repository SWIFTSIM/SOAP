**The velocity dispersion matrix** is defined as

.. math::

   V_{\rm{}disp,comp} = \frac{1}{\sum_{i={\rm{}comp}} m_i} \sum_{i={\rm{}comp}} m_i \vec{v}_{{\rm{}comp},r,i}\vec{v}_{{\rm{}comp},r,i},

where we compute the relative velocity as before, i.e. w.r.t. the centre of mass velocity of the particular
component of interest. While it is strictly speaking a :math:`3 \times 3` matrix, there are only 6 independent
components. We use the following convention to output those 6 components as a 6-element array:

.. math::

   V'_{\rm{}disp} = \begin{pmatrix}
   V_{xx} & V_{yy} & V_{zz} & V_{xy} & V_{xz} & V_{yz}
   \end{pmatrix}.

Other velocity dispersion definitions can be derived from this general form. The one-dimensional velocity dispersion can be calculated as

.. math::

   \sigma = \sqrt{\frac{V_{xx} + V_{yy} + V_{zz}}{3}}
