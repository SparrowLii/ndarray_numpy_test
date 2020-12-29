use ndarray::*;
use numpy::*;
use pyo3::{
    prelude::*,
    types::IntoPyDict,
    PyResult,
};
fn main()  {
    rayon_test();
    let _=numpy_test();
}
fn numpy_test()-> PyResult<()> {
    Python::with_gil(|py| {
        let np = py.import("numpy")?;
        let locals = [("np", np)].into_py_dict(py);
        let np_start=std::time::SystemTime::now();
        let _: &PyArray1<f64> = py
            .eval("np.sum(np.arange(0,64000000,1, dtype='float64').reshape(4000,-1),axis=1)", Some(locals), None)?
            .extract()?;
        let np_end=std::time::SystemTime::now();
        println!("np:{:?}",np_end.duration_since(np_start));

        Ok(())
    })
}
fn rayon_test(){
    use ndarray::parallel::prelude::*;

    let a = Array::range(0., 64000000., 1.).into_shape((4000, 16000)).unwrap();

    let rayon_start=std::time::SystemTime::now();
    let _:Vec<_>=a.axis_iter(Axis(0))
        .into_par_iter()
        .map(|row| row.sum())
        .collect();
    let rayon_end=std::time::SystemTime::now();
    println!("rayon:{:?}",rayon_end.duration_since(rayon_start));

    let no_rayon_start=std::time::SystemTime::now();
    let _:Vec<_>=a.axis_iter(Axis(0))
        .map(|row| row.sum())
        .collect();
    let no_rayon_end=std::time::SystemTime::now();
    println!("no_rayon:{:?}",no_rayon_end.duration_since(no_rayon_start));

}

