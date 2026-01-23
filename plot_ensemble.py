import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import argparse

# Header:
# number samples, inner min ave, inner min std  inner max ave  inner max std  outer min ave  outer min std  outer max ave  outer max std


def read_ensemble(infile):
    df = pd.read_csv(infile, skiprows=3, delimiter=',')
    return df


def stat_analysis(df):

    # calc inner. 
    inner_min = np.mean(df['inner min ave'])
    inner_max = np.mean(df['inner max ave'])

    inner_interval = (inner_min, inner_max)
    # calc inner. 
    outer_min = np.mean(df['outer min ave'])
    outer_max = np.mean(df['outer max ave'])

    outer_interval = (outer_min, outer_max)

    return inner_interval, outer_interval


def plot_converge(df, a=0.2):
    fig, axes = plt.subplots(2,2, sharex=True)

    fig.suptitle("Convergence of Mean w/ increased samples")

    # plot inner ellipsek
    axl, axr = axes[0, :]
    axl.errorbar(df['number samples'], df['inner min ave'], 
                 yerr=df['inner min std']/np.sqrt(df['number samples']),
                 ecolor=colors.to_rgba('tab:blue', a))
    axr.errorbar(df['number samples'], df['inner max ave'], 
                 yerr=df['inner max std']/np.sqrt(df['number samples']),
                 ecolor=colors.to_rgba('tab:blue', a))

    axl.set_ylabel("inner min ave")
    axr.set_ylabel("inner max ave")

    # plot outer ellipse
    axl, axr = axes[1, :]
    axl.errorbar(df['number samples'], df['outer min ave'], color='tab:orange',
                 yerr=df['outer min std']/np.sqrt(df['number samples']),
                 ecolor=colors.to_rgba('tab:orange', a))
    axr.errorbar(df['number samples'], df['outer max ave'], color='tab:orange',
                 yerr=df['outer max std']/np.sqrt(df['number samples']),
                 ecolor=colors.to_rgba('tab:orange', a))

    axl.set_ylabel("outer min ave")
    axr.set_ylabel("outer max ave")
    axl.set_xlabel("N samples")
    axr.set_xlabel("N samples")


    fig.tight_layout()
    return fig


def plot_ensemble_hist(df, Nbins=20):
    fig, axes = plt.subplots(2,2)

    # plot inner ellipse
    axl, axr = axes[0, :]
    axl.hist(df['inner min'], bins=Nbins)
    axr.hist(df['inner max'], bins=Nbins)

    axl.set_ylabel("inner min")
    axr.set_ylabel("inner max")

    # plot outer ellipse
    axl, axr = axes[1, :]
    axl.hist(df['outer min'], color='tab:orange', bins=Nbins)
    axr.hist(df['outer max'], color='tab:orange', bins=Nbins)

    axl.set_ylabel("outer min")
    axr.set_ylabel("outer max")

    fig.tight_layout()

    return fig


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str,
                        help='Name of the input csv file containing inner_min, inner_max etc. ensemble results')
    parser.add_argument('infile_conv', type=str,
                        help='Name of the input csv file containing ensemble convergance analsys results')
    parser.add_argument('-o', '--outfile', type=str, default='plot_ensemble_hist.png',
                        help='Name of the output plot file simple histograms')
    parser.add_argument('-oc', '--outfile_conv', type=str, default='plot_ensemble_conv.png',
                        help='Name of the output plot file for convergance analysis')
    parser.add_argument('-N', '--Nbins', type=float, default=20,
                        help='Number of bins to use in histogram plot')
    parser.add_argument('-a', '--alpha', type=float, default=0.2,
                        help='alpha value of the errobars in convergance analysis')
    args = parser.parse_args()

    # red in data
    df = read_ensemble(args.infile)
    df_conv = read_ensemble(args.infile_conv)

    # plot the data
    fig = plot_ensemble_hist(df, Nbins=args.Nbins)
    fig_conv = plot_converge(df_conv, a=args.alpha)

    # save the data
    fig.savefig(args.outfile)
    fig_conv.savefig(args.outfile_conv)
