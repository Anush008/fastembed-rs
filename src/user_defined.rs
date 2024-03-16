use reqwest;
use std::io::{self, Error, Read};
use std::{
    fmt::Display,
    fs::{read_dir, File},
    path::{Path, PathBuf},
    thread::available_parallelism,
};





pub trait FileSource {
    fn read_file(&self) -> Result<String, Error>;
}

impl FileSource for PathBuf {
    fn read_file(&self) -> Result<String, Error> {
        let mut file = File::open(self)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        Ok(contents)
    }
}

impl FileSource for String {
    fn read_file(&self) -> Result<String, Error> {
        let response = reqwest::blocking::get(self).unwrap();
        if response.status().is_success() {
            Ok(response.text().unwrap())
        } else {
            Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Failed to fetch URL: {}", response.status()),
            ))
        }
    }
}
