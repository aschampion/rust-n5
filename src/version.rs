/// A semantic version.
///
/// # Examples
///
/// ```
/// # use n5::Version;
/// # use std::str::FromStr;
/// let v = Version::from_str("1.2.3-suffix").unwrap();
///
/// assert_eq!(v.get_major(), 1);
/// assert_eq!(v.get_minor(), 2);
/// assert_eq!(v.get_patch(), 3);
/// assert_eq!(v.get_suffix(), "-suffix");
/// assert_eq!(v.to_string(), "1.2.3-suffix");
///
/// assert!(v.is_compatible(&Version::from_str("1.1").unwrap()));
/// assert!(!v.is_compatible(&Version::from_str("2.1").unwrap()));
/// ```
#[derive(Debug, Eq, PartialEq)]
pub struct Version {
    major: i32,
    minor: i32,
    patch: i32,
    suffix: String,
}

impl Version {
    pub fn new(major: i32, minor: i32, patch: i32, suffix: &str) -> Version {
        Version {
            major,
            minor,
            patch,
            suffix: suffix.to_owned(),
        }
    }

    pub fn get_major(&self) -> i32 {
        self.major
    }

    pub fn get_minor(&self) -> i32 {
        self.minor
    }

    pub fn get_patch(&self) -> i32 {
        self.patch
    }

    pub fn get_suffix(&self) -> &str {
        &self.suffix
    }

    pub fn is_compatible(&self, other: &Version) -> bool {
        other.get_major() <= self.major
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}{}", self.major, self.minor, self.patch, self.suffix)
    }
}

impl std::str::FromStr for Version {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let re = regex::Regex::new(r"(\d+)(\.(\d+))?(\.(\d+))?(.*)").unwrap();
        Ok(match re.captures(s) {
            Some(caps) => {
                Version {
                    major: caps.get(1).and_then(|m| m.as_str().parse().ok()).unwrap_or(0),
                    minor: caps.get(3).and_then(|m| m.as_str().parse().ok()).unwrap_or(0),
                    patch: caps.get(5).and_then(|m| m.as_str().parse().ok()).unwrap_or(0),
                    suffix: caps.get(6).map_or("", |m| m.as_str()).to_owned(),
                }
            }
            None => Version {
                major: 0,
                minor: 0,
                patch: 0,
                suffix: "".into(),
            }
        })
    }
}
