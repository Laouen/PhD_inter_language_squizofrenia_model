from tqdm import tqdm

def chained_tqdm(totals, descriptions):
    pbars = []
    for total in totals:
        pbars.append(tqdm(total=total))
        pbars[-1].refresh()
        pbars[-1].reset()
        if len(descriptions) >= len(pbars):
            pbars[-1].set_description(descriptions[len(pbars)-1])
    return pbars
