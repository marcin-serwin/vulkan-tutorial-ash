use std::ffi::{c_char, CStr};

// based on https://users.rust-lang.org/t/can-i-conveniently-compile-bytes-into-a-rust-program-with-a-specific-alignment/24049/2
macro_rules! include_bytes_as_array {
    ($align_ty:ty, $path:expr) => {{
        #[repr(C)] // guarantee 'bytes' comes after '_align'
        struct AlignedAs<Align, Bytes: ?Sized> {
            pub _align: [Align; 0],
            pub bytes: Bytes,
        }

        // this assignment is made possible by CoerceUnsized
        static ALIGNED: &AlignedAs<$align_ty, [u8]> = &AlignedAs {
            _align: [],
            bytes: *include_bytes!($path),
        };

        let bytes_len = ALIGNED.bytes.len();
        let ty_size = ::std::mem::size_of::<$align_ty>();

        debug_assert!(bytes_len % ty_size == 0);

        unsafe {
            ::std::slice::from_raw_parts::<'static, $align_ty>(
                ALIGNED.bytes.as_ptr().cast(),
                bytes_len / ty_size,
            )
        }
    }};
}

pub(crate) use include_bytes_as_array;

macro_rules! cstr {
    ($str:literal) => {{
        const RESULT: &CStr = unsafe {
            ::std::ffi::CStr::from_bytes_with_nul_unchecked(concat!($str, "\0").as_bytes())
        };
        RESULT
    }};
}

pub(crate) use cstr;

macro_rules! offset_of {
    ($ty:ident, $field:ident) => {{
        const OFFSET: usize = {
            let s = ::std::mem::MaybeUninit::<$ty>::uninit();
            let s_ptr = s.as_ptr();
            let f_ptr = unsafe { ::std::ptr::addr_of!((*s_ptr).$field) };
            (unsafe { f_ptr.byte_offset_from(s_ptr) }) as usize
        };
        OFFSET
    }};
}

pub(crate) use offset_of;

pub fn name_to_cstr(name: &[c_char]) -> &CStr {
    CStr::from_bytes_until_nul(unsafe { &*(name as *const [c_char] as *const [u8]) }).unwrap()
}
