import setuptools
 
package_name = 'gace'

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as req:
    requirements = req.read().splitlines()
 
setuptools.setup( name                          = package_name
                , version                       = '0.1.0'
                , author                        = 'Yannick Uhlmann'
                , author_email                  = 'augustunderground@protonmail.com'
                , description                   = 'Analog circuit design gym environments for operational amplifiers'
                , long_description              = long_description
                , long_description_content_type = 'text/markdown'
                , url                           = 'https://github.com/augustunderground/cyrcus'
                , packages                      = setuptools.find_packages()
                , classifiers                   = [ 'Development Status :: 2 :: Pre-Alpha'
                                                  , 'Programming Language :: Python :: 3'
                                                  , 'Operating System :: POSIX :: Linux' ]
                , python_requires               = '>=3.8'
                , install_requires              = requirements
                , package_data                  = { '': ['*.hy', '__pycache__/*'] }
#                                                        , 'models/*', 'lib/*' ] }
                , include_package_data          = True
                , )
