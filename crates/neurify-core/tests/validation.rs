use neurify_core::{compile, Arch, AttnMode, ErrorKind, Graph, Kind, Options};

fn constant_graph() -> Graph {
    let mut g = Graph::new();
    let value = g.constant(7.0, Kind::Int).unwrap();
    g.output(value).unwrap();
    g
}

#[test]
fn rejects_invalid_numeric_options() {
    let g = constant_graph();
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.0] {
        let options = Options {
            margin: value,
            ..Options::default()
        };
        let error = compile(&g, &options).unwrap_err();
        assert_eq!(error.kind, ErrorKind::Invalid);
        assert!(error.msg.contains("margin"));
    }
    for value in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        let options = Options {
            slope: value,
            ..Options::default()
        };
        assert!(compile(&g, &options).unwrap_err().msg.contains("slope"));
    }
    for value in [f64::NAN, f64::INFINITY, -1.0, 0.0, 1.0, 2.0] {
        let options = Options {
            arch: Arch::Transformer,
            attn_tol: value,
            ..Options::default()
        };
        assert!(compile(&g, &options).unwrap_err().msg.contains("attn_tol"));
        let options = Options {
            arch: Arch::Transformer,
            layernorm: true,
            ln_tol: value,
            ..Options::default()
        };
        assert!(compile(&g, &options).unwrap_err().msg.contains("ln_tol"));
    }
}

#[test]
fn unused_transformer_tolerances_do_not_reject_exact_architectures() {
    let g = constant_graph();
    let options = Options {
        attn_tol: 0.0,
        ln_tol: 0.0,
        ..Options::default()
    };
    assert!(compile(&g, &options).unwrap().1.exact());
    for attention in [AttnMode::Hard, AttnMode::Relu] {
        let options = Options {
            arch: Arch::Transformer,
            attention,
            ..options.clone()
        };
        assert!(compile(&g, &options).unwrap().1.exact());
    }
}

#[test]
fn table_rejects_overflowing_shapes_and_invalid_origins() {
    let mut g = Graph::new();
    let x = g.input("x", Kind::Int, 0.0, 1.0).unwrap();
    let error = g
        .table(
            vec![x, x],
            vec![0.0, 0.0],
            vec![usize::MAX, 2],
            vec![0.0],
            Kind::Int,
        )
        .unwrap_err();
    assert_eq!(error.kind, ErrorKind::Invalid);
    for origin in [f64::NAN, f64::INFINITY, 0.5] {
        assert!(g
            .table(vec![x], vec![origin], vec![2], vec![0.0, 1.0], Kind::Int)
            .is_err());
    }
    let table = g
        .table(vec![x], vec![0.0], vec![2], vec![2.0, 3.0], Kind::Int)
        .unwrap();
    g.output(table).unwrap();
    assert_eq!(g.eval(&[0.0]).unwrap(), vec![2.0]);
    assert_eq!(g.eval(&[1.0]).unwrap(), vec![3.0]);
}

#[test]
fn zero_input_networks_evaluate_once_in_a_batch() {
    for arch in [Arch::Mlp, Arch::ResMlp, Arch::Glu, Arch::Transformer] {
        let (net, _) = compile(
            &constant_graph(),
            &Options {
                arch,
                ..Options::default()
            },
        )
        .unwrap();
        assert_eq!(net.forward_batch(&[]), vec![7.0]);
    }
}

#[test]
#[should_panic(expected = "batch must contain complete input rows")]
fn rejects_incomplete_batch_rows() {
    let mut g = Graph::new();
    let x = g.input("x", Kind::Int, 0.0, 1.0).unwrap();
    g.input("y", Kind::Int, 0.0, 1.0).unwrap();
    g.output(x).unwrap();
    let (net, _) = compile(&g, &Options::default()).unwrap();
    net.forward_batch(&[1.0]);
}

#[test]
fn batches_preserve_all_complete_rows_and_allow_empty_batches() {
    let mut g = Graph::new();
    let x = g.input("x", Kind::Int, 0.0, 1.0).unwrap();
    g.input("y", Kind::Int, 0.0, 1.0).unwrap();
    g.output(x).unwrap();
    let (net, _) = compile(&g, &Options::default()).unwrap();
    assert!(net.forward_batch(&[]).is_empty());
    assert_eq!(net.forward_batch(&[0.0, 1.0, 1.0, 0.0]), vec![0.0, 1.0]);
}

#[test]
#[should_panic(expected = "a zero-input network expects no input values")]
fn zero_input_network_rejects_supplied_values() {
    let (net, _) = compile(&constant_graph(), &Options::default()).unwrap();
    net.forward_batch(&[1.0]);
}

#[test]
#[should_panic(expected = "expected exactly n_in input values")]
fn single_forward_rejects_extra_inputs() {
    let (net, _) = compile(&constant_graph(), &Options::default()).unwrap();
    net.forward(&[1.0]);
}
