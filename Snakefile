rule initial_setup:
    output: "src/data/variables.asdf"
    conda: "environment.yml"
    cache: True
    script: "src/scripts/setup.py"


rule simulate_p2ls:
    input:
        "src/data/variables.asdf"
    output: "src/data/P2LS.asdf"
    params:
        n_sprp = 15,
        n_z_center = 1_000,
        n_z_lr = 1_000
    conda: "environment.yml"
    cache: True
    script: "src/scripts/p2ls/simulate_p2ls.py"


rule sample_p2ls:
    input:
        "src/data/variables.asdf",
        "src/data/P2LS.asdf"
    output: "src/data/p2ls_samples.npy"
    params:
        seed = 42
    conda: "environment.yml"
    cache: True
    script: "src/scripts/p2ls/sample_p2ls.py"


rule compute_signal:
    input:
        "src/data/variables.asdf",
        "src/data/P2LS.asdf",
        "src/data/p2ls_samples.npy",
    output: "src/tex/output/signal.txt"
    params:
        seed = 99
    conda: "environment.yml"
    cache: True
    script: "src/scripts/signal/compute_signal.py"


rule compute_undamped_signal:
    input:
        "src/data/variables.asdf",
        "src/data/P2LS.asdf",
        "src/data/p2ls_samples.npy"
    output: "src/tex/output/undamped_signal.txt"
    params:
        seed = 99
    conda: "environment.yml"
    cache: True
    script: "src/scripts/signal/compute_undamped_signal.py"


rule compute_signal_ratio:
    input:
        "src/tex/output/signal.txt",
        "src/tex/output/undamped_signal.txt",
    output: "src/tex/output/signal_ratio.txt"
    conda: "environment.yml"
    cache: True
    script: "src/scripts/signal/compute_ratio.py"


rule compute_correlation_EE:
    input:
        "src/data/variables.asdf",
        "src/data/P2LS.asdf",
        "src/data/p2ls_samples.npy"
    output: "src/data/C_EE_corr.npy"
    params:
        seed = 67
    conda: "environment.yml"
    cache: True
    script: "src/scripts/correlation/compute_ee.py"


rule compute_correlation_EL:
    input:
        "src/data/variables.asdf",
        "src/data/P2LS.asdf",
        "src/data/p2ls_samples.npy"
    output: "src/data/C_EL_corr.npy"
    params:
        seed = 78
    conda: "environment.yml"
    cache: True
    script: "src/scripts/correlation/compute_el.py"


rule compute_correlation_LL:
    input:
        "src/data/variables.asdf",
        "src/data/P2LS.asdf",
        "src/data/p2ls_samples.npy"
    output: "src/data/C_LL_corr.npy"
    params:
        seed = 89
    conda: "environment.yml"
    cache: True
    script: "src/scripts/correlation/compute_ll.py"


rule compute_cls:
    input:
        "src/data/C_EE_corr.npy",
        "src/data/C_EL_corr.npy",
        "src/data/C_LL_corr.npy"
    output: "src/data/cls.npy"
    conda: "environment.yml"
    cache: True
    script: "src/scripts/cls/compute_cls.py"


rule compute_EE_0:
    input:
        "src/data/C_EE_corr.npy"
    output: "src/tex/output/C_EE_zero.txt"
    conda: "environment.yml"
    cache: True
    script: "src/scripts/detectability/ee_0.py"


rule compute_EE_1:
    input:
        "src/data/C_EE_corr.npy"
    output: "src/tex/output/C_EE_one.txt"
    conda: "environment.yml"
    cache: True
    script: "src/scripts/detectability/ee_1.py"


rule compute_EE_ratio:
    input:
        "src/data/C_EE_corr.npy"
    output: "src/tex/output/C_EE_ratio.txt"
    conda: "environment.yml"
    cache: True
    script: "src/scripts/detectability/ratio_ee.py"


rule compute_EL_0:
    input:
        "src/data/variables.asdf",
        "src/data/C_EL_corr.npy"
    output: "src/tex/output/C_EL_zero.txt"
    conda: "environment.yml"
    cache: True
    script: "src/scripts/detectability/el_0.py"


rule compute_EL_1:
    input:
        "src/data/variables.asdf",
        "src/data/C_EL_corr.npy"
    output: "src/tex/output/C_EL_one.txt"
    conda: "environment.yml"
    cache: True
    script: "src/scripts/detectability/el_1.py"


rule compute_EL_ratio:
    input:
        "src/data/C_EL_corr.npy"
    output: "src/tex/output/C_EL_ratio.txt"
    conda: "environment.yml"
    cache: True
    script: "src/scripts/detectability/ratio_el.py"


rule compute_LL_0:
    input:
        "src/data/C_LL_corr.npy"
    output: "src/tex/output/C_LL_zero.txt"
    conda: "environment.yml"
    cache: True
    script: "src/scripts/detectability/ll_0.py"


rule sn_pixie:
    input:
        "src/data/variables.asdf",
        "src/data/C_EL_corr.npy"
    output: "src/tex/output/sn_pixie.txt"
    conda: "environment.yml"
    cache: True
    script: "src/scripts/detectability/sn_pixie.py"


rule std_error_pixie:
    input:
        "src/data/C_EL_corr.npy"
    output: "src/tex/output/std_error_pixie.txt"
    conda: "environment.yml"
    cache: True
    script: "src/scripts/detectability/std_error_pixie.py"


rule sn_act:
    input:
        "src/data/variables.asdf",
        "src/data/C_EL_corr.npy"
    output: "src/tex/output/sn_act.txt"
    conda: "environment.yml"
    cache: True
    script: "src/scripts/detectability/sn_act.py"


rule sn_s4:
    input:
        "src/data/variables.asdf",
        "src/data/C_EL_corr.npy"
    output: "src/tex/output/sn_s4.txt"
    conda: "environment.yml"
    cache: True
    script: "src/scripts/detectability/sn_s4.py"
