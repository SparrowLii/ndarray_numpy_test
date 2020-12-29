use ndarray::*;
use numpy::*;
use pyo3::{
    prelude::*,
    types::IntoPyDict,
    PyResult,
};
macro_rules! assert_approx_eq {
    ($x: expr, $y: expr) => {
        assert!(($x - $y) <= std::f64::EPSILON);
    };
}
fn main()  {
    test1();
    let _=test2()?;
    Python::with_gil(|py| {
        let np = py.import("numpy")?;
        let locals = [("np", np)].into_py_dict(py);
        let pyarray: &PyArray2<f64> = py
            .eval("np.absolute(np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype='float64'))", Some(locals), None)?
            .extract()?;
        let iter = NpySingleIterBuilder::readonly(pyarray.readonly()).build()?;

        // The order of iteration is not specified, so we should restrict ourselves
        // to tests that don't verify a given order.
        assert_approx_eq!(iter.sum::<f64>(), 15.0);
        Ok(())
    })
}
fn test2()-> PyResult<()> {
    Python::with_gil(|py| {
        let np = py.import("numpy")?;
        let locals = [("np", np)].into_py_dict(py);
        let np_start=std::time::SystemTime::now();
        let pyarray: &PyArray1<f64> = py
            .eval("np.sum(np.arange(0,640000000,1, dtype='float64').reshape(4000,-1),axis=1)", Some(locals), None)?
            .extract()?;
        let np_end=std::time::SystemTime::now();
        println!("np:{:?}",np_end.duration_since(np_start));
        let iter = NpySingleIterBuilder::readonly(pyarray.readonly()).build()?;
        let c:Vec<_>=iter.collect();
        //println!("c:{:?}",c);

        // The order of iteration is not specified, so we should restrict ourselves
        // to tests that don't verify a given order.
        Ok(())
    })
}
fn test1(){
    use ndarray::parallel::prelude::*;

    let a = Array::range(0., 64000000., 1.).into_shape((4000, 16000)).unwrap();

    let rayon_start=std::time::SystemTime::now();
    let _:Vec<_>=a.axis_iter(Axis(0))
        .into_par_iter()
        .map(|row| row.sum())
        .collect();
    let rayon_end=std::time::SystemTime::now();
    println!("rayon:{:?}",rayon_end.duration_since(rayon_start));
    println!("rayon:{:?}",sums);

    let no_rayon_start=std::time::SystemTime::now();
    let sums2:Vec<_>=a.axis_iter(Axis(0))
        .map(|row| row.sum())
        .collect();
    let no_rayon_end=std::time::SystemTime::now();
    println!("no_rayon:{:?}",no_rayon_end.duration_since(no_rayon_start));*/
    //println!("no_rayon:{:?}",sums2);

    let a = arr2(&[[1., 2.],
        [3., 4.]]);
    assert_eq!(a.sum(), 10.);

    let a = arr2(&[[1., 2., 3.],
        [4., 5., 6.]]);
    assert!(
        a.sum_axis(Axis(0)) == aview1(&[5., 7., 9.]) &&
            a.sum_axis(Axis(1)) == aview1(&[6., 15.]) &&

            a.sum_axis(Axis(0)).sum_axis(Axis(0)) == aview0(&21.)
    );


}

