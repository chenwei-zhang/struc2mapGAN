from argparse import ArgumentParser, Namespace
import os
import sys
import subprocess
from inference import inference 
from StructureBlurrer import pdb2vol
from molmap import molmap



def parse_arguments():
    parser = ArgumentParser(description='Generate experimental-like high-resolution cryo-EM density maps from PDB files, using a pretreined GAN.')

    parser.add_argument("--pdb", required=True, 
                        help="Path to the input PDB or CIF file.")
    parser.add_argument("--output_mrc", required=True, 
                        help="Path and filename to save the output MRC file.")
    parser.add_argument("--ref_map", default=False, 
                        help="Path to a reference map in MRC format")
    parser.add_argument("--chimerax", type=str, default='/usr/bin/chimerax',
                        help="Path to the ChimeraX executable.")
    parser.add_argument('--ckpt', type=str, required=True, 
                        help='Path to the model checkpoint')    
    parser.add_argument("--res", type=float, default=2.0, 
                        help="Resolution of the output map.")
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=1, 
                        help='Number of workers for dataloader')
    parser.add_argument("--mode", default='pdb2vol', choices=['molmap', 'pdb2vol'], 
                        help="physical simulaton mode converting pdb to volume")
    parser.add_argument("-s", "--sigma_coeff", type=float, default=0.356, 
                        help="Sigma coefficient for blurring.")
    parser.add_argument("-r", "--real_space", action="store_true", default=False,
                        help="Whether to perform real-space blurring.")
    parser.add_argument("-n", "--normalize", default=True,
                        help="Whether to normalize the output map.")
    parser.add_argument("-bb", "--backbone_only", action="store_true", default=False,
                        help="Whether to only consider backbone atoms.")
    parser.add_argument("-b", "--bin_mask", action="store_true", default=False,
                        help="Whether to binarize the output map.")
    parser.add_argument("-c", "--contour", type=float, default=0.0, 
                        help="Contour level for contouring the output map.")
    parser.add_argument("--verbose", action="store_true", default=False, 
                        help="Print verbose output.")
    parser.add_argument("--rm_sim", action="store_true", default=False,
                        help="Remove the physical simulated map.")
    args = parser.parse_args()
    
    return args


def resample(chimerax, ref_map, sim_map, gan_map, sim_map_resample, gan_map_resample, verbose=False):
    result = subprocess.run([chimerax, '--nogui', 
                            '--cmd', 
                            f'open {ref_map}; \
                            open {sim_map}; \
                            open {gan_map}; \
                            vol #1 #2 #3 step 1 ; \
                            vol resample #2 onGrid #1 gridStep 1; \
                            vol resample #3 onGrid #1 gridStep 1; \
                            save {sim_map_resample} #4; \
                            save {gan_map_resample} #5; \
                            exit'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)
    if verbose:
        print(result.stdout)
        sys.stdout.flush()
        
    print(f'Resampled maps saved to {sim_map_resample} and {gan_map_resample}')
        
        
        
def mrcGAN(args: Namespace):
    # Convert pdb to map by physical simulation
    
    if args.mode == 'pdb2vol':
        pdb2vol(
            input_pdb=args.pdb,
            resolution=args.res,
            output_mrc=args.output_mrc,
            ref_map=False, 
            sigma_coeff=args.sigma_coeff,
            real_space=args.real_space,
            normalize=args.normalize,
            backbone_only=args.backbone_only,
            contour=args.contour,
            bin_mask=args.bin_mask,
            return_data=False,
        )
        sim_map = f'{args.output_mrc}_pdb2vol.mrc'
        
    elif args.mode == 'molmap':
        molmap(
            pdb=args.pdb,
            output_mrc=args.output_mrc,
            res=args.res,
            normalize=args.normalize,
            verbose=args.verbose,
        )
        sim_map = f'{args.output_mrc}_molmap.mrc'

    # GAN inference
    inference(
        sim_map,
        args.ckpt,
        args.batch_size,
        args.num_workers,
        save_dir=os.path.dirname(args.output_mrc),
        )
    
    directory = os.path.dirname(sim_map)
    sim_map_name = os.path.basename(sim_map).split('.')[0]
    gan_map_name = f'{sim_map_name}_gan'
    
    if args.rm_sim:
        os.remove(sim_map)
    
    return directory, sim_map_name, gan_map_name



if __name__ == "__main__":
    
    args = parse_arguments()
    
    directory, sim_map_name, gan_map_name = mrcGAN(args)
    
    if args.ref_map is not False:
        
        # reshape GAN maps
        sim_map = os.path.join(directory, f'{sim_map_name}.mrc')
        gan_map = os.path.join(directory, f'{gan_map_name}.mrc')
        sim_map_resample = os.path.join(directory, f'{sim_map_name}_resample.mrc')
        gan_map_resample = os.path.join(directory, f'{gan_map_name}_resample.mrc')
        
        print('Resampling maps...')
        
        # reshape GAN maps
        resample(args.chimerax, args.ref_map, sim_map, gan_map, sim_map_resample, gan_map_resample, args.verbose)

        # # # reshape experimental maps
        # ref_map = args.ref_map
        # ref_map_resample = os.path.join(directory, f'{os.path.basename(ref_map).split(".")[0]}_resample.mrc')
        
        # subprocess.run([args.chimerax, '--nogui', 
        #                 '--cmd', 
        #                 f'open {gan_map}; \
        #                 open {ref_map}; \
        #                 vol #1 #2 step 1 ; \
        #                 vol resample #2 onGrid #1 gridStep 1; \
        #                 save {ref_map_resample} #3; \
        #                 exit'],
        #                 stdout=subprocess.PIPE,
        #                 stderr=subprocess.PIPE,
        #                 text=True)
        
        # print('Resampling maps...')
        # print(f'Resampled reference map saved to {ref_map_resample}')
    