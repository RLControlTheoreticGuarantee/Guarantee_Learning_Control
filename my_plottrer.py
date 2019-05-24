import pandas as pd
import os
import os.path as osp
import numpy as np
import ENV
import matplotlib.pyplot as plt
ROOT_DIR = './log'


COLORS = ['blue',  'red','green', 'brown','salmon','cyan', 'magenta','darkred',  'yellow', 'black', 'purple', 'pink',
          'teal',  'lightblue', 'orange', 'lavender', 'turquoise','lime',
        'darkgreen', 'tan',  'gold']

COLORS_map = {'ppo2':'blue', 'ppo2_lyapunov':'green', 'SAC_lyapunov':'red', 'SAC':'brown',
          'LAC':'red',
          'CPO':'blue','CPO_lyapunov':'green',}


CONTENT_YLABEL = ['return', 'safety cost']

label_fontsize = 10
tick_fontsize = 14
linewidth = 3
markersize = 10

def read_csv(fname):
    return pd.read_csv(fname, index_col=None, comment='#')

def load_results(args,alg_list, contents, env,rootdir=ROOT_DIR):
    # if isinstance(rootdir, str):
    #     rootdirs = [osp.expanduser(rootdir)]
    # else:
    #     dirs = [osp.expanduser(d) for d in rootdir]
    results = {}
    for name in env:
        results[name] = {}
    exp_dirs = os.listdir(rootdir)

    for exp_dir in exp_dirs:

        if exp_dir in env:

            exp_path = os.path.join(rootdir, exp_dir)
            alg_dirs = os.listdir(exp_path)
            for alg_dir in alg_dirs:
                plot = False
                for alg in alg_list:
                    if alg in alg_dir:
                        plot = True
                if plot:
                    alg_path = os.path.join(exp_path, alg_dir)
                    trial_dirs = os.listdir(alg_path)
                    trials = []
                    result={}
                    min_length = 1e10
                    for trial_dir in trial_dirs:
                        if trial_dir not in args['plot_list']:
                            continue
                        full_path = os.path.join(alg_path, trial_dir)
                        try:
                            trials.append(read_csv(full_path + '/progress.csv'))
                        except pd.errors.EmptyDataError:
                            continue
                        serial_length = len(trials[-1][contents[0]])
                        if serial_length<min_length:
                            min_length = serial_length
                    for key in contents:
                        try:
                            summary = [trial[key][:min_length] for trial in trials]
                        except KeyError:
                            continue

                        result[key] = np.mean(summary, axis=0)
                        std = np.std(summary, axis=0)
                        result[key+'max'] = result[key] + std
                        result[key + 'min'] = result[key] - std
                    try:
                        result['total_timesteps'] = trials[0]['total_timesteps'][:min_length]/1000
                    except IndexError:
                        print('index error')
                    results[exp_dir][alg_dir] = result


    return results

def plot_results(results,alg_list, contents, figsize = None):
    if not args['formal_plot']:
        nrows = len(contents)
        ncols = len(results)
        figsize = figsize or (6, 6)
        f, axarr = plt.subplots(nrows, ncols, sharex=False, squeeze=False, figsize=figsize)

    # plot ep rewards
    content_index = 0
    for content in contents:
        exp_index = 0
        for exp in results.keys():
            min_length = 1e10
            if not args['formal_plot']:
                ax = axarr[content_index][exp_index]
            else:
                fig = plt.figure(figsize=(9, 6))
                ax = fig.add_subplot(111)
            for alg in results[exp].keys():
                result = results[exp][alg]
                color_index = list(results[exp].keys()).index(alg)
                try:
                    length= result['total_timesteps'].values[-1]
                except KeyError:
                    continue
                if args['formal_plot'] and alg in COLORS_map.keys():
                    color = COLORS_map[alg]
                else:
                    color = COLORS[color_index]
                try:
                    ax.plot(result['total_timesteps'], result[content], color=color, label=alg)
                except KeyError:
                    continue
                ax.fill_between(result['total_timesteps'], result[content +'min'],result[content+'max'],
                                color=color, alpha=.25)
                if length<min_length:
                    min_length = length
            # if exp_index==0:
                # plt.ylabel(CONTENT_YLABEL[content_index],fontsize=label_fontsize)

            ax.legend(fontsize=12, loc=2,fancybox=False, shadow=False)
            plt.xticks(fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            ax.grid(True)
            if 'ylim' in args.keys():
                plt.ylim(args['ylim'][0], args['ylim'][-1])

            # fig = plt.gcf()
            ax.set_title(exp, fontsize=label_fontsize)
            # fig.set_size_inches(9, 6)
            if args['formal_plot']:
                plt.xlim(0, round(min_length/1000,1)*1000)
                plt.show()
            exp_index +=1
        content_index +=1

    if not args['formal_plot']:
        f.set_size_inches(9, 6)
        plt.show()

    return
ENV_LIST = ['CartPolecons-v0',
            'CartPolecost-v0',
            'Antcons-v0',
            'HalfCheetahcons-v0',
            'PongNoFrameskip-v5',
            'Pointcircle-v0',
            'FetchReach-v1',
            'Quadrotorcons-v0',
            'Quadrotorcost-v0'
            'Carcost-v0']

PLOT_CONTENT = ['eprewmean', 'eplrewmean','violation_times','eplenmean','lyapunov_lambda', 'eval_eprewmean', 'eval_eplrewmean']
ALG_LIST = ['ppo2', 'SAC','CPO', 'LAC']
# ALG_LIST = ['SAC_lyapunov-cons-1', 'SAC_lyapunov-cons-2','SAC_lyapunov-cons-3', 'SAC_lyapunov-cons-4']
def main(args,alg_list = ALG_LIST, content=PLOT_CONTENT,env=ENV_LIST):

    results = load_results(args,alg_list, content, env)
    plot_results(results,alg_list, content)
if __name__ == '__main__':

    args = {'plot_list':[str(i) for i in range(0,1)],
            'formal_plot':False,
            # 'ylim':[0,50],
            }
    # args = {'lim': [str(i) for i in range(0, 10)]}
    # args = {'plot_list': [str(i) for i in range(1)]}
    alg_list = ALG_LIST[0:4]
    content = PLOT_CONTENT[0:2]
    env = ENV_LIST[0:1]
    main( args, alg_list, content, env)