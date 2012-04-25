class apt-update {
  exec { "/usr/bin/apt-get -y update": }
}

class python {
  package {
    ["build-essential", "python", "python-dev", "python-setuptools"]:
      ensure => present;
  }
  exec { "easy_install pip":
    path => "/usr/local/bin:/usr/bin:/bin",
    refreshonly => true,
    require => Package["python-setuptools"],
    subscribe => Package["python-setuptools"];
  }
}

class perl {
  package {
    "perl": ensure => present;
  }
}

class python-libs {
  # ubuntu packages required to build the following python packages
  package {
    ["libatlas-base-dev", "gfortran", "libpng12-dev", "python-gtk2-dev",
     "libfreetype6-dev", "libgraphviz-dev"]:
       ensure => present;
  }
  # python packages
  package {
    "numpy":
      ensure => "1.6.1",
      provider => "pip";
    "scipy":
      ensure => "0.10.1",
      provider => "pip",
      require => [
                  Package["gfortran"],
                  Package["libatlas-base-dev"],
                  Package["numpy"],
                 ];
    "sympy":
      ensure => "0.7.1",
      provider => "pip";
    "matplotlib":
      ensure => "1.1.0",
      provider => "pip",
      require => [
                  Package["libpng12-dev"],
                  Package["python-gtk2-dev"],
                  Package["libfreetype6-dev"],
                  Package["numpy"]
                 ];
    "ipython":
      ensure => "0.12",
      provider => "pip";
    "pygraphviz":
      ensure => "1.1",
      provider => "pip",
      require => Package["libgraphviz-dev"];
  }
}

class bionetgen {
  package { "wget": ensure => present; }
  exec { "/usr/bin/wget https://github.com/downloads/jmuhlich/bionetgen/BioNetGen_2.1.8_rev597.tgz":
    cwd => "/tmp",
    creates => "/tmp/BioNetGen_2.1.8_rev597.tgz",
    require => Package["wget"];
  }
  file { "/tmp/BioNetGen_2.1.8_rev597.tgz": }
  exec { "/bin/tar xzf /tmp/BioNetGen_2.1.8_rev597.tgz":
    cwd => "/home/demo",
    user => demo,
    group => demo,
    creates => "/home/demo/BioNetGen",
    require => [
                File["/tmp/BioNetGen_2.1.8_rev597.tgz"],
                File["/home/demo"],
                ],
  }
  file { "/usr/local/share/BioNetGen":
    ensure => link,
    target => "/home/demo/BioNetGen";
  }
}

class pysb {
  package {
    "pysb":
      provider => "pip";
  }
  user { "demo": ensure => present }
  file { "/home/demo":
    ensure => directory,
    source => "/etc/skel",
    recurse => true,
    owner => "demo",
    group => "demo";
  }
}

class puppet-misc {
  group { "puppet": ensure => present; }
}

class { "apt-update": }
class { "python": }
class { "python-libs": }
class { "perl": }
class { "bionetgen": }
class { "pysb": }
class { "puppet-misc": }

Class["python"] -> Package <| provider == pip |>
Class["apt-update"] -> Package <| provider == apt |>
