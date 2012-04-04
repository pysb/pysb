class puppet-misc {
    group { "puppet": ensure => "present"; }
}

class apt-update {
    exec { "/usr/bin/apt-get -y update": }
}

class python {
    package {
        "build-essential": ensure => latest;
        "python": ensure => latest;
        "python-dev": ensure => latest;
        "python-setuptools": ensure => latest;
    }
    exec { "easy_install pip":
        path => "/usr/local/bin:/usr/bin:/bin",
        refreshonly => true,
        require => Package["python-setuptools"],
        subscribe => Package["python-setuptools"];
    }
}

class pysb {
    # ubuntu packages required to build the following python packages
    package {
        "libatlas-base-dev":
            ensure => "latest";
        "gfortran":
            ensure => "latest";
        "libpng12-dev":
            ensure => "latest";
        "python-gtk2-dev":
            ensure => "latest";
        "libfreetype6-dev":
            ensure => "latest";
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
                Package["numpy"]
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
                Package["numpy"],
            ];
        "ipython":
            ensure => "0.12",
            provider => "pip";

# TODO
#        "pysb":
#            provider => "pip";
    }
}

stage { "pre": before => Stage["main"] }

class { "apt-update": stage => "pre" }
class { "python": }
class { "pysb": }
class { "puppet-misc": }

Class["python"] -> Package <| provider == pip |>
