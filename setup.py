import os

from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name,
                  module,
                  sources,
                  cuda_extra_args=[],
                  cpp_extra_args=[],
                  extra_include_path=[]):

    if not torch.cuda.is_available():
        raise EnvironmentError('CUDA is required to compile this package!')

    extra_compile_args = {'cxx': cpp_extra_args + [],
                          'nvcc': cuda_extra_args + ['-D__CUDA_NO_HALF_OPERATORS__',
                                                     '-D__CUDA_NO_HALF_CONVERSIONS__',
                                                     '-D__CUDA_NO_HALF2_OPERATORS__']}

    return CUDAExtension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        extra_compile_args=extra_compile_args,
        include_dirs=extra_include_path,
        define_macros=[]  # use extra_agrs instead
    )


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        list[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


if __name__ == '__main__':
    setup(
        name='mdet',
        version='0.0.1',
        description='todo',
        long_description='todo',
        author='todo',
        author_email='todo',
        keywords='computer vision, object detection',
        url='todo',
        package_dir={"": "."},
        packages=find_packages(where='.'),
        include_package_data=True,
        package_data={'mdet.ops': ['*/*.so']},
        classifiers=[],
        license='',
        requires=[],
        ext_modules=[
            make_cuda_ext(
                name='voxelization',
                module='mdet.ops.voxelization',
                sources=[
                    'src/voxelization.cpp',
                    'src/voxelization_cpu.cpp',
                    'src/voxelization_cuda.cu',
                ]),
            make_cuda_ext(
                name='iou3d',
                module='mdet.ops.iou3d',
                sources=[
                    'src/iou3d.cpp',
                    'src/iou3d_cpu.cpp',
                    'src/iou3d_cuda.cu',
                    'src/nms_bev_kernel.cu',
                ]),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
