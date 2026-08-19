use anyhow::{anyhow, Context, Result};
use v4l::buffer::Type;
use v4l::io::traits::OutputStream;
use v4l::prelude::*;
use v4l::video::Output;
use v4l::FourCC;

use crate::cli::Mode;

pub trait Sink {
    fn write_frame(&mut self, data: &[u8]) -> Result<()>;
}

/// Alpha needs a format that carries it, which only OBS consumes. Everything
/// else emits YUYV so browsers, Zoom and Discord work.
pub fn fourcc_for(mode: Mode) -> FourCC {
    match mode {
        Mode::Alpha => FourCC::new(b"AR24"),
        Mode::Greenscreen | Mode::Image => FourCC::new(b"YUYV"),
    }
}

pub fn bytes_per_pixel(mode: Mode) -> usize {
    match mode {
        Mode::Alpha => 4,
        Mode::Greenscreen | Mode::Image => 2,
    }
}

pub struct V4lSink {
    stream: MmapStream<'static>,
    frame_len: usize,
}

impl V4lSink {
    pub fn open(path: &str, width: u32, height: u32, mode: Mode) -> Result<V4lSink> {
        let dev = Device::with_path(path).with_context(|| {
            format!(
                "opening {path}. If it does not exist, create one with:\n  \
                 sudo modprobe v4l2loopback devices=1 video_nr=9 card_label=Matting exclusive_caps=1"
            )
        })?;

        let mut fmt = Output::format(&dev).context("querying output format")?;
        fmt.width = width;
        fmt.height = height;
        fmt.fourcc = fourcc_for(mode);
        // v4l2loopback only sets its buffer length once a format is applied,
        // so this call must happen before allocating buffers.
        let fmt = Output::set_format(&dev, &fmt).context("setting output format")?;

        if fmt.width != width || fmt.height != height || fmt.fourcc != fourcc_for(mode) {
            // v4l2loopback pins its format for as long as a consumer has the
            // device open, so switching modes while OBS (or a browser, or
            // ffplay) is watching fails here rather than at startup.
            return Err(anyhow!(
                "{path} would not accept {} {width}x{height}; it is currently {} {}x{}.\n\
                 This usually means something else still has the device open — a virtual \
                 camera keeps its pixel format while a consumer is attached.\n\
                 Close whatever is viewing it (check with `fuser -v {path}`), then try again.",
                fourcc_for(mode),
                fmt.fourcc,
                fmt.width,
                fmt.height
            ));
        }

        let dev: &'static Device = Box::leak(Box::new(dev));
        let stream = MmapStream::with_buffers(dev, Type::VideoOutput, 4)
            .context("allocating output buffers")?;

        Ok(V4lSink {
            stream,
            frame_len: width as usize * height as usize * bytes_per_pixel(mode),
        })
    }
}

impl Sink for V4lSink {
    fn write_frame(&mut self, data: &[u8]) -> Result<()> {
        if data.len() != self.frame_len {
            return Err(anyhow!(
                "expected {} bytes, got {}",
                self.frame_len,
                data.len()
            ));
        }
        let (buf, meta) = OutputStream::next(&mut self.stream).context("dequeuing output buffer")?;
        buf[..data.len()].copy_from_slice(data);
        meta.field = 0;
        meta.bytesused = data.len() as u32;
        Ok(())
    }
}

/// Records frames in memory so the pipeline can be tested without a device.
pub struct TestSink {
    frames: Vec<Vec<u8>>,
}

impl TestSink {
    pub fn new() -> TestSink {
        TestSink { frames: Vec::new() }
    }

    pub fn frames(&self) -> &[Vec<u8>] {
        &self.frames
    }
}

impl Default for TestSink {
    fn default() -> Self {
        Self::new()
    }
}

impl Sink for TestSink {
    fn write_frame(&mut self, data: &[u8]) -> Result<()> {
        self.frames.push(data.to_vec());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::Mode;

    #[test]
    fn alpha_mode_uses_bgra_fourcc() {
        assert_eq!(fourcc_for(Mode::Alpha), v4l::FourCC::new(b"AR24"));
        assert_eq!(bytes_per_pixel(Mode::Alpha), 4);
    }

    #[test]
    fn opaque_modes_use_yuyv() {
        for mode in [Mode::Greenscreen, Mode::Image] {
            assert_eq!(fourcc_for(mode), v4l::FourCC::new(b"YUYV"));
            assert_eq!(bytes_per_pixel(mode), 2);
        }
    }

    #[test]
    fn test_sink_records_frames() {
        let mut sink = TestSink::new();
        sink.write_frame(&[1, 2, 3]).unwrap();
        sink.write_frame(&[4, 5, 6]).unwrap();
        assert_eq!(sink.frames(), &[vec![1, 2, 3], vec![4, 5, 6]]);
    }

    #[test]
    fn opening_a_missing_device_is_an_error() {
        assert!(V4lSink::open("/dev/definitely-not-a-loopback", 1024, 576, Mode::Alpha).is_err());
    }
}
