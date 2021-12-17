# Copyright 2021 The Scenic Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""setup.py for Scenic.

Install for development:

  pip intall -e . .[tests]
"""

from distutils import cmd
import os
import urllib.request

from setuptools import find_packages
from setuptools import setup
from setuptools.command import install

SIMCLR_DIR = "simclr/tf2"
DATA_UTILS_URL = "https://raw.githubusercontent.com/google-research/simclr/master/tf2/data_util.py"


class DownloadSimCLRAugmentationCommand(cmd.Command):
  """Downloads SimCLR data_utils.py as it's not built into an egg."""
  description = __doc__
  user_options = []

  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def run(self):
    build_cmd = self.get_finalized_command("build")
    dist_root = os.path.realpath(build_cmd.build_lib)
    output_dir = os.path.join(dist_root, SIMCLR_DIR)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "data_util.py")
    downloader = urllib.request.URLopener()
    downloader.retrieve(DATA_UTILS_URL, output_path)


TFMODEL_DIR = "tensorflow_models/official/vision/image_classification"
TFMODEL_DATA_UTILS_URL = "https://raw.githubusercontent.com/tensorflow/models/master/official/vision/image_classification/augment.py"


class DownloadTFModelAugmentationCommand(cmd.Command):
  """Downloads TF model vision augment.py as it's not built into an egg."""
  description = __doc__
  user_options = []

  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def run(self):
    build_cmd = self.get_finalized_command("build")
    dist_root = os.path.realpath(build_cmd.build_lib)
    output_dir = os.path.join(dist_root, TFMODEL_DIR)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "augment.py")
    downloader = urllib.request.URLopener()
    downloader.retrieve(TFMODEL_DATA_UTILS_URL, output_path)


class InstallCommand(install.install):

  def run(self):
    self.run_command("simclr_download")
    self.run_command("tfmodel_download")
    install.install.run(self)

tests_require = [
    "pytest",
]

setup(
    name="scenic",
    version="0.0.1",
    description=("A Jax Library for Computer Vision Research and Beyond."),
    author="Scenic Authors",
    author_email="no-reply@google.com",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="http://github.com/google-research/scenic",
    license="Apache 2.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "absl-py",
        "jax",
        "flax",
        "ml-collections",
        "tensorflow",
        "tfds-nightly",
        "ott-jax",
        "immutabledict",
        "numpy",
        "clu",
        "sklearn",
        "seaborn",
        "tqdm",
        "pycocotools",
        "dmvr @ git+git://github.com/deepmind/dmvr",
    ],
    cmdclass={
        "simclr_download": DownloadSimCLRAugmentationCommand,
        "tfmodel_download": DownloadTFModelAugmentationCommand,
        "install": InstallCommand,
    },
    tests_require=tests_require,
    extras_require=dict(test=tests_require),
    classifiers=[
        "Development Status :: 1 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="Scenic",
)
