"""Explicit, separately identifiable cube/gripper and cube/bin materials."""


def apply_materials(env,params):
    for model in env.models.values():
        cube=model.geom("cube").id
        model.geom_friction[cube,0]=params["cube_sliding"]
        model.geom_friction[cube,1]=params["cube_torsional"]
        if params.get("bin_sliding") is not None:
            for i in range(5):
                geom=model.geom(f"bin_{i}").id
                # Bin supplies this contact's material. Preserve the solver
                # parameters previously supplied by the higher-priority cube.
                model.geom_priority[geom]=int(model.geom_priority[cube])+1
                model.geom_condim[geom]=model.geom_condim[cube]
                model.geom_solref[geom]=model.geom_solref[cube]
                model.geom_solimp[geom]=model.geom_solimp[cube]
                model.geom_friction[geom]=[params["bin_sliding"],params["bin_torsional"],model.geom_friction[cube,2]]

