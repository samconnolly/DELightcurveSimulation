#!/usr/bin/env python

if __name__ == '__main__':
    from distutils.core import setup
    setup(  name="DELCgen",
            version="0.3alpha",
            description="DELCgen: Simulating Lightcurves",
            author="Sam Connolly",
            author_email="sdc1g08@soton.ac.uk",
            url="https://github.com/samconnolly/DELightcurveSimulation",
            license="Academic Free License",
            classifiers=[
                'Development Status :: 5 - Production/Stable',
                'Environment :: Console',
                'Operating System :: OS Independent',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: Academic Free License (AFL)',
                'Programming Language :: Python',
                'Topic :: Scientific/Engineering',
                 ],
            requires=['NumPy (>=1.3)',],
            long_description="""
            Python implementation of the light curve simulation algorithm from 
            Emmanoulopoulos et al., 2013, Monthly Notices of the Royal 
            Astronomical Society, 433, 907. Produces lightcurves with
            a given PSD and PDF.
            """,
            py_modules=["DELCgen"]
            )
