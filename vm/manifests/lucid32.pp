stage { "pre": before => Stage["main"] }
class python {
    package {
        "build-essential": ensure => latest;
        "python": ensure => "2.6.5-0ubuntu1";
        "python-dev": ensure => "2.6.5-0ubuntu1";
        "python-setuptools": ensure => installed;
    }
    exec { "easy_install pip":
        path => "/usr/local/bin:/usr/bin:/bin",
        refreshonly => true,
        require => Package["python-setuptools"],
        subscribe => Package["python-setuptools"],
    }
}
class { "python": stage => "pre" }


class pysb {
    package {
        "libatlas-base-dev":
            ensure => "latest";
        "gfortran":
            ensure => "latest";
        "libpng-dev":
            ensure => "latest";
        "python-gtk2-dev":
            ensure => "latest";
        "libfreetype6-dev":
            ensure => "latest";
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
                Package["libpng-dev"],
                Package["python-gtk2-dev"],
                Package["libfreetype6-dev"]
            ];
        "ipython":
            ensure => "0.12",
            provider => "pip";

# TODO
#        "pysb":
#            provider => "pip";
    }
}
class { "pysb": }